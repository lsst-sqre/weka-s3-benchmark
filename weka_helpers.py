"""Helper functions and classes for Weka benchmarking."""
import datetime, json, os, random, shutil, statistics, subprocess

from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Self

from lsst.daf.butler import Butler

__all__ = ["Dataset", "PrimitiveResultMap", "ResultMap", "convert_results_to_primitive", "get_results"]

type PrimitiveResultMapItem = dict[str, str|int | float | None]

@dataclass
class ResultMapItem:
    """Results for a single HTTP object."""
    url: str
    size: int = 0
    initial_start_time: datetime.datetime | None = None
    initial_end_time: datetime.datetime | None = None
    initial_elapsed_time: datetime.timedelta | float | None = None
    recheck_start_time: datetime.datetime | None = None
    recheck_end_time: datetime.datetime | None = None
    recheck_elapsed_time: datetime.timedelta | float | None = None
    initial_speed_reported: float  = 0.0
    recheck_speed_reported: float | None = None
    initial_speed_measured: float  = 0.0
    recheck_speed_measured: float | None = None
    ratio_reported: float | None = None  # Initial/recheck
    ratio_measured: float | None = None  # Initial/recheck

    def calculate_ratio(self) -> None:
        if self.initial_speed_reported and self.recheck_speed_reported:
            self.ratio_reported = self.recheck_speed_reported/self.initial_speed_reported
        if self.initial_speed_measured and self.recheck_speed_measured:
            self.ratio_measured = self.recheck_speed_measured/self.initial_speed_measured

    def to_primitive(self) -> PrimitiveResultMapItem:
        initial_start_time: datetime.datetime | None = None
        initial_end_time: datetime.datetime | None = None
        recheck_start_time: datetime.datetime | None = None
        recheck_end_time: datetime.datetime | None = None
        initial_elapsed_time: datetime.timedelta | None = None
        recheck_elapsed_time: datetime.timedelta | None = None
        if self.initial_start_time is not None:
            initial_start_time = self.initial_start_time.isoformat()
        if self.initial_end_time is not None:
            initial_end_time = self.initial_end_time.isoformat()
        if self.recheck_start_time is not None:
            recheck_start_time = self.recheck_start_time.isoformat()
        if self.recheck_end_time is not None:
            recheck_end_time = self.recheck_end_time.isoformat()
        if self.initial_elapsed_time is not None:
            initial_elapsed_time = self.initial_elapsed_time.total_seconds()
        if self.recheck_elapsed_time is not None:
            recheck_elapsed_time = self.recheck_elapsed_time.total_seconds()
        return {
            "url": self.url,
            "size": self.size,
            "initial_start_time": initial_start_time,
            "initial_end_time": initial_end_time, 
            "recheck_start_time": recheck_start_time,
            "recheck_end_time": recheck_end_time,
            "initial_elapsed_time": initial_elapsed_time,
            "initial_speed_reported": self.initial_speed_reported,
            "recheck_elapsed_time": recheck_elapsed_time,
            "recheck_speed_reported": self.recheck_speed_reported,
            "ratio_reported": self.ratio_reported,
            "initial_speed_measured": self.initial_speed_measured,
            "recheck_speed_measured": self.recheck_speed_measured,
            "ratio_measured": self.ratio_measured,
        }

    @classmethod
    def from_primitive(cls, obj: PrimitiveResultMapItem) -> Self:
        st_i_s = obj["initial_start_time"]
        et_i_s = obj["initial_end_time"]
        st_r_s = obj["recheck_start_time"]
        et_r_s = obj["recheck_end_time"]
        initial_start_time = datetime.datetime.fromisoformat(st_i_s) if st_i_s else None
        initial_end_time = datetime.datetime.fromisoformat(et_i_s) if et_i_s else None
        recheck_start_time = datetime.datetime.fromisoformat(st_r_s) if st_r_s else None
        recheck_end_time = datetime.datetime.fromisoformat(et_r_s) if et_r_s else None       
        initial_elapsed_time = initial_end_time - initial_start_time if (initial_end_time and initial_start_time) else None
        recheck_elapsed_time = recheck_start_time - recheck_start_time if (recheck_end_time and recheck_start_time) else None
        return(cls(
            url = obj["url"],
            size = obj["size"],
            initial_start_time = initial_start_time,
            initial_end_time = initial_end_time,
            recheck_start_time = recheck_start_time,
            recheck_end_time = recheck_end_time,
            initial_elapsed_time = initial_elapsed_time,
            recheck_elapsed_time = recheck_elapsed_time,
            initial_speed_reported = obj["initial_speed_reported"],
            recheck_speed_reported = obj["recheck_speed_reported"],
            ratio_reported = obj["ratio_reported"],            
            initial_speed_measured = obj["initial_speed_measured"],
            recheck_speed_measured = obj["recheck_speed_measured"],
            ratio_measured = obj["ratio_measured"],
        ))

    def to_json(self) -> str:
        return json.dumps(self.to_primitive(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, j_obj: str) -> Self:
        return cls.from_primitive(json.loads(j_obj))
        

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

    def to_primitive(self):
        return asdict(self)

    @classmethod
    def from_primitive(cls, data: list[float]) -> Self:
        return cls(_data=data)

    def to_json(self) -> str:
        return json.dumps(self.to_primitive(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, j_obj: str) -> Self:
        return cls.from_primitive(json.loads(j_obj))
            
type PrimitiveResultMap = dict[str,str|dict[str,str|float|int|None]]      

@dataclass
class ResultMap:
    """Results for a particular dataset."""
    display_name: str
    dataset_name: str
    count: int = 0
    limit: int = 0
    result_items: list[ResultMapItem] = field(default_factory=list)
    object_size_statistics: ResultStatistics | None = None
    initial_statistics_measured: ResultStatistics | None = None
    initial_statistics_reported: ResultStatistics | None = None
    recheck_statistics_reported: ResultStatistics | None = None
    ratio_statistics_measured: ResultStatistics | None = None
    ratio_statistics_reported: ResultStatistics | None = None

    @property
    def sizes(self) -> list[int]:
        return [x.size for x in self.result_items]

    @property
    def initial_speeds_measured(self) -> list[float]:
        return [x.initial_speed_measured for x in self.result_items]

    @property
    def initial_speeds_reported(self) -> list[float]:
        return [x.initial_speed_reported for x in self.result_items]
        
    @property
    def recheck_speeds_measured(self) -> list[float | None]:
        return [x.recheck_speed_measured for x in self.result_items]

    @property
    def recheck_speeds_reported(self) -> list[float | None]:
        return [x.recheck_speed_reported for x in self.result_items]

    @property
    def ratios_measured(self) -> list[float | None]:
        return [x.ratio_measured for x in self.result_items]

    @property
    def ratios_reported(self) -> list[float | None]:
        return [x.ratio_reported for x in self.result_items]

    def calculate_ratios(self) -> None:
        for it in self.result_items:
            it.calculate_ratio()
    
    def calculate_size_statistics(self) -> None:
        """Statistics for HTTP object size."""
        self.object_size_statistics=ResultStatistics(
            _data=self.sizes
        )

    def calculate_initial_statistics(self) -> None:
        """Statistics for HTTP initial fetch speed."""
        self.initial_statistics_reported=ResultStatistics(
            _data=self.initial_speeds_reported
        )
        self.initial_statistics_measured=ResultStatistics(
            _data=self.initial_speeds_measured
        )

    def calculate_recheck_statistics(self) -> None:
        """Statistics for HTTP recheck fetch speed."""
        self.recheck_statistics_reported=ResultStatistics(
            _data=[x for x in self.recheck_speeds_reported if x is not None]
        )
        self.recheck_statistics_measured=ResultStatistics(
            _data=[x for x in self.recheck_speeds_measured if x is not None]
        )

    def calculate_ratio_statistics(self) -> None:
        """Statistics for recheck/initial throughput."""
        self.ratio_statistics_reported=ResultStatistics(
            _data=[x for x in self.ratios_reported if x is not None]
        )
        self.ratio_statistics_measured=ResultStatistics(
            _data=[x for x in self.ratios_measured if x is not None]
        )
    
    def to_primitive(self) -> PrimitiveResultMapItem:
        return {
            "display_name": self.display_name,
            "dataset_name": self.dataset_name,
            "count": self.count,
            "limit": self.limit,
            "result_items": [ x.to_primitive() for x in self.result_items ],
            "object_size_statistics": self.object_size_statistics.to_primitive(),
            "initial_statistics_measured": self.initial_statistics_measured.to_primitive(),
            "initial_statistics_reported": self.initial_statistics_reported.to_primitive(),
            "recheck_statistics_measured": self.recheck_statistics_measured.to_primitive() if self.recheck_statistics_measured else None,
            "recheck_statistics_reported": self.recheck_statistics_measured.to_primitive() if self.recheck_statistics_reported else None,
            "ratio_statistics_measured": self.ratio_statistics_measured.to_primitive() if self.ratio_statistics_measured else None,
            "ratio_statistics_reported": self.ratio_statistics_reported.to_primitive() if self.ratio_statistics_reported else None,
        }

    @classmethod
    def from_primitive(cls, obj: PrimitiveResultMap) -> Self:
        res_items=[ResultMapItem.from_primitive(x) for x in obj["result_items"]]
        retval = cls(
            display_name=obj["display_name"],
            dataset_name=obj["dataset_name"],
            count=obj["count"],
            limit=obj["limit"],
            result_items=res_items,
        )
        retval.calculate_ratios()
        retval.calculate_size_statistics()
        retval.calculate_initial_statistics()
        retval.calculate_recheck_statistics()
        retval.calculate_ratio_statistics()
        return retval

    def to_json(self) -> str:
        return json.dumps(self.to_primitive(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, j_obj: str) -> Self:
        return cls.from_primitive(json.loads(j_obj))
    
@dataclass
class Dataset:
    """Dataset: from ``limit`` rows, select ``count`` results."""
    display_name: str
    dataset_name: str
    count: int
    limit: int

def now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)
    
def get_result_path(datadir: Path) -> Path:
    """Return path to JSON file holding results."""
    nnow = now()
    of_name = f"weka-gateway-{nnow.year}.{nnow.month:02d}.{nnow.day:02d}.{nnow.hour:02d}.{nnow.minute:02d}.{nnow.second:02d}-{os.getenv("USER")}.json"
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
        result_map = ResultMap(
            display_name=name,
            dataset_name=dsname,
            count=count,
            limit=limit
        )
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
    i_s = "initial" if initial else "recheck"
    for idx, item in enumerate(result_map.result_items):
            print(f"Getting {i_s} speed for {dataset}: {idx+1}/{tot}...", end='')
            st = now()
            speed_reported = find_speed(item.url)
            et = now()
            if initial:
                item.initial_start_time_initial = st
                item.initial_speed_reported = speed_reported
                item.initial_end_time = et
                item.initial_elapsed_time = et - st
                item.initial_speed_measured = item.size / item.initial_elapsed_time.total_seconds()
            else:
                item.recheck_start_time = st
                item.recheck_speed_reported = speed_reported
                item.recheck_end_time = et
                item.recheck_elapsed_time = et - st
                item.recheck_speed_measured = item.size / item.recheck_elapsed_time.total_seconds()
            print(f"{speed_reported} bytes/s")
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
    for obj in d_obj:
        res.append(ResultMap.from_primitive(obj))
    return res

def convert_results_to_primitive(res: list[ResultMap]) -> list[PrimitiveResultMap]:
    """Convert results to JSON-representable form."""
    retval: list[PrimitiveResultMap] = []
    for rm in res:
        retval.append(rm.to_primitive())
    return retval
        
def write_results(
    *,
    butler_config: str,
    butler_collections: str,
    data_map: list[Dataset],
    output_dir: Path|None = None,
    recheck_speeds: bool = True,
) -> list[ResultMap]:
    """Get results and write them to a file.  Return results.
    """
    res: list[ResultMap] = []
    if output_dir is None:
        output_dir = Path(".")
    result_path=get_result_path(output_dir)
    res = build_results(butler_config, butler_collections, data_map)
    for result_map in res:
        result_map.start_time = datetime.datetime.now(tz=datetime.UTC)
        result_map = get_all_sizes(result_map)
        result_map = find_all_speeds(result_map, initial=True)
        result_map.calculate_initial_statistics()
        if recheck_speeds:
            result_map = find_all_speeds(result_map, initial=False)
            result_map.calculate_ratios()
            result_map.calculate_recheck_statistics()
            result_map.calculate_ratio_statistics()
    outp = convert_results_to_primitive(res)
    result_path.write_text(json.dumps(outp))
    return res