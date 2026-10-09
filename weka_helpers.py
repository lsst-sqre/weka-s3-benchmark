"""Helper functions and classes for Weka benchmarking."""
import datetime, json, os, random, shutil, statistics, subprocess

from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path

from lsst.daf.butler import Butler

__all__ = ["Dataset", "convert_results_to_primitive", "get_results"]

@dataclass
class ResultMapItem:
    """Results for a single HTTP object."""
    url: str
    size: int = 0
    initial_speed: float  = 0.0
    recheck_speed: float | None = None
    ratio: float | None = None  # Initial/recheck

    def calculate_ratio(self) -> None:
        if self.initial_speed and self.recheck_speed:
            self.ratio = self.recheck_speed/self.initial_speed

@dataclass
class ResultStatistics:
    """Statistics for a particular list of values."""
    _data: InitVar[list[float]]
    CV: float | None = None
    mean: float | None = None
    median: float | None = None
    pvariance: float | None = None
    pstdev: float | None = None

    def __post_init__(self,_data:list[float]) -> None:
        if _data and len(_data) > 0:
            self.mean = statistics.mean(_data)
            self.median = statistics.median(_data)
            self.pstdev = statistics.pstdev(_data)
            self.pvariance = statistics.pvariance(_data)
            self.CV = self.pstdev / self.mean
            
        

@dataclass
class ResultMap:
    """Results for a particular dataset."""
    display_name: str
    dataset_name: str
    initial_statistics: ResultStatistics | None = None
    object_size_statistics: ResultStatistics | None = None
    recheck_statistics: ResultStatistics | None = None
    ratio_statistics: ResultStatistics | None = None
    result_items: list[ResultMapItem] = field(default_factory=list)

    
    def calculate_ratios(self) -> None:
        """Ratio of rechecked speed to initial speed."""
        for item in self.result_items:
            item.calculate_ratio()

    @property
    def sizes(self) -> list[int]:
        return [x.size for x in self.result_items]

    @property
    def initial_speeds(self) -> list[float]:
        return [x.initial_speed for x in self.result_items]

    @property
    def recheck_speeds(self) -> list[float | None]:
        return [x.recheck_speed for x in self.result_items]

    @property
    def ratios(self) -> list[float | None]:
        return [x.ratio for x in self.result_items]

    def calculate_size_statistics(self) -> None:
        """Statistics for HTTP object size."""
        self.object_size_statistics=ResultStatistics(
            _data=self.sizes
        )

    def calculate_initial_statistics(self) -> None:
        """Statistics for HTTP initial fetch speed."""
        self.initial_statistics=ResultStatistics(
            _data=self.initial_speeds
        )

    def calculate_recheck_statistics(self) -> None:
        """Statistics for HTTP recheck fetch speed."""
        self.recheck_statistics=ResultStatistics(
            _data=[x for x in self.recheck_speeds if x is not None]
        )

    def calculate_ratio_statistics(self) -> None:
        """Statistics for recheck/initial speed ratios."""
        self.ratio_statistics=ResultStatistics(
            _data=[x for x in self.ratios if x is not None]
        )

type PrimitiveResultMap = dict[str,str|dict[str,str|float|int|None]]
    
@dataclass
class Dataset:
    """Dataset: from ``limit`` rows, select ``count`` results."""
    display_name: str
    dataset_name: str
    count: int
    limit: int

def get_result_path(datadir: Path) -> Path:
    """Return path to JSON file holding results."""
    now = datetime.datetime.now(tz=datetime.UTC)
    of_name = f"weka-gateway-{now.year}.{now.month:02d}.{now.day:02d}.{now.hour:02d}.{now.minute:02d}.{now.second:02d}-{os.getenv("USER")}.json"
    return datadir / of_name
    
def build_results(cfg: str, collections: str, data_map: list[Dataset]) -> list[ResultMap]:
    """From a list of parameters, create a random sample of URLs to benchmark.
    """
    results: list[ResultMap] = []
    butler = Butler(cfg, collections=collections)
    for ds in data_map:
        name = ds.display_name
        dsname = ds.dataset_name
        count = ds.count
        limit = ds.limit
        if ((count <= 0) or (limit <= 0)):
            continue
        result_map = ResultMap(display_name=name, dataset_name=dsname)
        results.append(result_map)
        print(f"Butler query of {name}({dsname}), limit={limit}...", end='')
        q_results = butler.query_datasets(dsname, find_first=False, limit = limit)
        records = count if len(q_results) > count else len(q_results)
        urls = [str(butler.getURI(x)) for x in random.sample(q_results, records)]
        print(f"sampling {records} records")
        for url in urls:
            result_map.result_items.append(ResultMapItem(url=url))
    return results

def get_curl() -> str:
    """Return path to ``curl`` executable."""
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("'curl' must be on path")
    return curl
    
def get_size(url: str) -> int:
    """Return size of HTTP-accessible object.
    The HEAD method is blocked, but a byte range will work."""
    curl=get_curl()
    proc = subprocess.run([curl, url, "-v", "-r", "0-0", "-o", "/dev/null"], capture_output=True)
    s_errl = proc.stderr.decode().split("\n")
    return int([x for x in s_errl if x.startswith("< content-range:")][0].split("/")[-1])

def t_to_spd(t_spd: str) -> float:
    """Convert speed in suffixed units to integer."""
    mega = 2 << 19
    kilo = 2 << 9
    if t_spd.endswith("M"):
        return float(t_spd[:-1]) * mega
    elif t_spd.endswith("k"):
        return float(t_spd[:-1]) * kilo
    else:
        return float(t_spd)

def find_speed(url: str) -> float:
    """Get speed of transfer.
    
    If we're slower than 50kB/s for 30 seconds, abort."""
    curl=get_curl()
    proc = subprocess.run([curl, url, "-Y", "50000", "-o", "/dev/null"], capture_output=True)
    try:
        t_spd = proc.stderr.decode().split("\n")[2].split()[-1]  # First two lines are headers
        spd = t_to_spd(t_spd)
    except (IndexError, ValueError) as exc:
        print(f"Error converting speed: {exc}")
        # We didn't get the output we expected; probably timeout.
        spd=0
    return spd

def get_all_sizes(result_map: ResultMap) -> ResultMap:
    """Updates result_map in place with size of each HTTP object."""
    dataset = result_map.display_name
    tot=len(result_map.result_items)
    for idx, item in enumerate(result_map.result_items):
        print(f"Getting size for {dataset}: {idx+1}/{tot}...", end='')
        item.size = get_size(item.url)
        print(f"{item.size} bytes")
    result_map.calculate_size_statistics()
    return result_map

def find_all_speeds(result_map: ResultMap, *, initial=True) -> ResultMap:
    """Updates result_map in place with speed of each HTTP fetch."""
    dataset = result_map.display_name
    tot=len(result_map.result_items)
    for idx, item in enumerate(result_map.result_items):
            print(f"Getting speed for {dataset}: {idx+1}/{tot}...", end='')
            speed = find_speed(item.url)
            if initial:
                item.initial_speed = speed
            else:
                item.recheck_speed = speed
            print(f"{speed} bytes/s")
    if initial:
        result_map.calculate_initial_statistics()
    else:
        result_map.calculate_recheck_statistics()
        result_map.calculate_ratio_statistics()
    return result_map

def load_results_from_json(filename: Path) -> list[ResultMap]:
    """Read results from a persisted JSON file."""
    res: list[ResultMap] = []
    d_obj = json.loads(filename.read_text())
    # Ugh, should have used Pydantic.
    for obj in d_obj:
        r_it: list[ResultMapItem] = []
        for ritem in obj["result_items"]:
            r_it.append(ResultMapItem(**ritem))
        rm=ResultMap(
            display_name=obj["display_name"],
            dataset_name=obj["dataset_name"],
            result_items=r_it
        )
        rm.calculate_initial_statistics()
        rm.calculate_size_statistics()
        rm.calculate_ratios()
        rm.calculate_recheck_statistics()
        rm.calculate_ratio_statistics()
        res.append(rm)
    return res

def convert_results_to_primitive(res: list[ResultMap]) -> list[PrimitiveResultMap]:
    """Convert results to JSON-representable form."""
    d_obj: list[PrimitiveResultMap] = []
    for result_map in res:
        d_obj.append(asdict(result_map))
    return d_obj

def write_results_to_json(res: list[ResultMap], filename: Path) -> None:
    filename.write_text(
        json.dumps(
            convert_results_to_primitive(
                res
            )
        )
    )
        
def get_results(
    *,
    butler_config: str,
    butler_collections: str,
    data_map: list[Dataset],
    output_dir: Path|None = None,
    input_filename: Path|None = None,
    recheck_speeds: bool = True,
) -> list[ResultMap]:
    """Either get results and write them to a file, or read
    results from a file; in either case, return the results.
    """
    res: list[ResultMap] = []
    if output_dir is None:
        output_dir = Path(".")
    if input_filename:
        res = load_results_from_json(input_filename)
    else:
        result_path=get_result_path(output_dir)
        res = build_results(butler_config, butler_collections, data_map)
        for result_map in res:
            result_map = get_all_sizes(result_map)
            result_map = find_all_speeds(result_map, initial=True)
            result_map.calculate_initial_statistics()
            if recheck_speeds:
                result_map = find_all_speeds(result_map, initial=False)
                result_map.calculate_ratios()
                result_map.calculate_recheck_statistics()
                result_map.calculate_ratio_statistics()
        write_results_to_json(res, result_path)    
    return res