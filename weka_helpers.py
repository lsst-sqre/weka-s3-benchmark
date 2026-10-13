"""Helper functions and classes for Weka benchmarking."""
import asyncio
import datetime
import json
import logging
import os
import random
import shutil
import statistics
import sys

from dataclasses import InitVar, asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Self

from lsst.daf.butler import Butler

__all__ = [
    "Benchmarker",
    "Dataset",
    "DatasetList",
    "PrimitiveRatioContainer",
    "PrimitiveResultMap",
    "PrimitiveResultMapItem",
    "PrimitiveResultStatistics",
    "RatioContainer",
    "ResultMap",
    "ResultMapItem",
    "ResultStatistics",
    "convert_results_to_primitive",
    "generate_results",
    "load_results_from_json",
]

################### Data types for speed testing ###################
type PrimitiveResultMapItem = dict[str, str|int | float | None]

@dataclass
class ResultMapItem:
    """Results for a single HTTP request."""
    url: str
    size: int = 0
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None
    elapsed_time: datetime.timedelta | float | None = None
    speed_reported: float  = 0.0
    speed_measured: float | None = None
    rm_speed_ratio: float | None = None

    def to_primitive(self) -> PrimitiveResultMapItem:
        start_time: datetime.datetime | None = None
        end_time: datetime.datetime | None = None
        elapsed_time: datetime.timedelta | None = None
        if self.start_time is not None:
            start_time = self.start_time.isoformat()
        if self.end_time is not None:
            end_time = self.end_time.isoformat()
        if self.elapsed_time is not None:
            elapsed_time = self.elapsed_time.total_seconds()
        return {
            "url": self.url,
            "size": self.size,
            "start_time": start_time,
            "end_time": end_time, 
            "elapsed_time": elapsed_time,
            "speed_reported": self.speed_reported,
            "speed_measured": self.speed_measured,
            "rm_speed_ratio": self.rm_speed_ratio
        }

    @classmethod
    def from_primitive(cls, obj: PrimitiveResultMapItem) -> Self:
        st_s = obj["start_time"]
        et_s = obj["end_time"]
        el_s = obj["elapsed_time"]
        start_time = datetime.datetime.fromisoformat(st_s) if st_s else None
        end_time = datetime.datetime.fromisoformat(et_s) if et_s else None
        elapsed_time = datetime.timedelta(seconds=el_s) if el_s else None
        return(cls(
            url = obj["url"],
            size = obj["size"],
            start_time = start_time,
            end_time = end_time,
            elapsed_time = elapsed_time,
            speed_reported = obj["speed_reported"],           
            speed_measured = obj["speed_measured"],
            rm_speed_ratio = obj["rm_speed_ratio"],
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

    def to_json(self) -> str:
        return json.dumps(self.to_primitive(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, j_obj: str) -> Self:
        return cls.from_primitive(json.loads(j_obj))

class Generation(StrEnum):
    INITIAL = "initial"
    RECHECK = "recheck"
    
type PrimitiveResultMap = dict[str,str|dict[str,str|float|int|None]]      

@dataclass
class ResultMap:
    """Results for a particular dataset."""
    display_name: str
    dataset_name: str
    count: int = 0
    limit: int = 0
    generation: Generation = Generation.INITIAL
    debug: bool = False
    result_items: list[ResultMapItem] = field(default_factory=list)
    object_size: ResultStatistics | None = None
    speed_measured: ResultStatistics | None = None
    speed_reported: ResultStatistics | None = None
    rm_speed_ratio: ResultStatistics | None = None

    @property
    def sizes(self) -> list[int]:
        return [x.size for x in self.result_items]

    @property
    def speeds_measured(self) -> list[float]:
        return [x.speed_measured for x in self.result_items]

    @property
    def speeds_reported(self) -> list[float]:
        return [x.speed_reported for x in self.result_items]

    @property
    def speeds_ratio(self) -> list[float]:
        return [x.rm_speed_ratio for x in self.result_items]
        
    @property
    def start_times(self) -> list[datetime.datetime|None]:
        return [x.start_time for x in self.result_items]

    @property
    def end_times(self) -> list[datetime.datetime|None]:
        return [x.end_time for x in self.result_items]

    @property
    def elapsed_times(self) -> list[datetime.timedelta|None]:
        return [x.elapsed_time for x in self.result_items]

    @property
    def urls(self) -> list[str]:
        return [x.url for x in self.result_items]
    
    def calculate_statistics(self) -> None:
        """Statistics for HTTP fetch speed and size."""
        self.object_size=ResultStatistics(
            _data=self.sizes
        )
        self.speed_measured=ResultStatistics(
            _data=self.speeds_measured
        )
        self.speed_reported=ResultStatistics(
            _data=self.speeds_reported
        )
        self.rm_speed_ratio=ResultStatistics(
            _data=self.speeds_ratio
        )
    
    def to_primitive(self) -> PrimitiveResultMapItem:
        return {
            "display_name": self.display_name,
            "dataset_name": self.dataset_name,
            "count": self.count,
            "limit": self.limit,
            "generation": self.generation,
            "result_items": [ x.to_primitive() for x in self.result_items ],
            "object_size": self.object_size.to_primitive(),
            "speed_measured": self.speed_measured.to_primitive(),
            "speed_reported": self.speed_reported.to_primitive(),
            "rm_speed_ratio": self.rm_speed_ratio.to_primitive()
        }

    @classmethod
    def from_primitive(cls, obj: PrimitiveResultMap) -> Self:
        res_items=[ResultMapItem.from_primitive(x) for x in obj["result_items"]]
        retval = cls(
            display_name=obj["display_name"],
            dataset_name=obj["dataset_name"],
            count=obj["count"],
            limit=obj["limit"],
            generation=Generation(obj["generation"]),
            result_items=res_items,
        )
        # Don't try to import statistics; just recalculate them.
        retval.calculate_statistics()
        return retval

    def to_json(self) -> str:
        return json.dumps(self.to_primitive(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, j_obj: str) -> Self:
        return cls.from_primitive(json.loads(j_obj))

type PrimitiveRatioContainer = dict[str,ResultMap|list[float]|ResultStatistics|None]

@dataclass
class RatioContainer:
    """This always holds at least one series of data.  It may hold two;
    we use this to resample a given set of URLs to investigate caching behavior."""
    display_name: str
    dataset_name: str
    count: int
    limit: int
    initial: ResultMap
    recheck: ResultMap | None = None
    ratios_measured: list[float] | None = None
    ratios_reported: list[float] | None = None
    ratio_statistics_measured: ResultStatistics | None = None
    ratio_statistics_reported: ResultStatistics | None = None

    def calculate_ratios(self) -> None:
        m_ratios: list[float] = []
        r_ratios: list[float] = []
        if not self.recheck or not self.recheck.result_items:
            return None
        self.validate_paired_resultmaps()
        for idx, i_itm in enumerate(self.initial.result_items):
            r_itm = self.recheck.result_items[idx]
            _have_results_check_result_map_items(i_itm, r_itm, idx)
            m_ratios.append(r_itm.speed_measured / i_itm.speed_measured)
            r_ratios.append(r_itm.speed_reported / i_itm.speed_reported)
        self.ratios_measured=m_ratios
        self.ratios_reported=r_ratios
        self.ratio_statistics_measured = ResultStatistics(_data=m_ratios)
        self.ratio_statistics_reported = ResultStatistics(_data=r_ratios)

    def validate_paired_resultmaps(self) -> None:
        if not self.recheck:
            raise ValueError("No recheck result map")
        if not self.recheck.result_items:
            raise ValueError("No recheck result items")
        iin = len(self.initial.result_items)
        rin = len(self.recheck.result_items)
        if iin != rin:
            raise ValueError(
                f"Initial map: {iin} items != {rin} for recheck map"
            )
        for idx, iit in enumerate(self.initial.result_items):
            rit = self.recheck.result_items[idx]
            _sanity_check_result_map_items(iit, rit)

    def _prepare_recheck_items(self) -> None:
        if not self.recheck:
            raise ValueError("No recheck result map")
        if not self.recheck.result_items:
            raise ValueError("No recheck result items")
        for idx, iit in enumerate(self.initial.result_items):
            rit = self.recheck.result_items[idx]
            rit.url = iit.url  # Copy, not reference
            rit.size = iit.size
    
    @classmethod
    def from_primitive(cls, obj: PrimitiveRatioContainer) -> Self:
        r_rm = ResultMap.from_primitive(obj["recheck"]) if obj["recheck"] else None
        retval = cls(
            display_name=obj["display_name"],
            dataset_name=obj["dataset_name"],
            count=obj["count"],
            limit=obj["limit"],
            initial=ResultMap.from_primitive(obj["initial"]),
            recheck=r_rm,
            ratios_measured=list(obj["ratios_measured"]),
            ratios_reported=list(obj["ratios_reported"]),
        )
        retval.calculate_ratios()
        return retval

    def to_primitive(self) -> PrimitiveRatioContainer:
        return({
            "display_name": self.display_name,
            "dataset_name": self.dataset_name,
            "count": self.count,
            "limit": self.limit,
            "initial": self.initial.to_primitive(),
            "recheck": self.recheck.to_primitive() if self.recheck else None,
            "ratios_measured": list(self.ratios_measured) if self.ratios_measured else None,
            "ratios_reported": list(self.ratios_reported) if self.ratios_reported else None,
            "ratio_statistics_measured": self.ratio_statistics_measured.to_primitive() if self.ratio_statistics_measured else None,
            "ratio_statistics_reported": self.ratio_statistics_reported.to_primitive() if self.ratio_statistics_reported else None,
        })

    def to_json(self) -> str:
        return json.dumps(self.to_primitive(), indent=2)
    
    @classmethod
    def from_json(cls, j_obj:str) -> Self:
        return cls(from_primitive(json.loads(j_obj)))

type ResultSet = list[RatioContainer]
type PrimitiveResultSet = list[PrimitiveRatioContainer]

# Configuration object class

@dataclass
class Dataset:
    """Object to characterize source of benchmark data.

    Parameters
    ----------
    display_name
        User-friendly name for dataset collection (e.g."big_datasets").
    dataset_name
        Internal (Butler) name for dataset collection (e.g. "run_provenance").
    count
        How many URLs to actually retrieve.
    limit
        How many URLs to get handles for from the butler.
    concurrency
        How many fetches to dispatch at once.
    """
    display_name: str
    dataset_name: str
    count:int = 0
    limit:int = 10000
    concurrency:int|None = None

type DatasetList = list[Dataset]

################### End of data types for speed testing ###################
    
#### Private utility functions

def _now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)

def _cpus() -> int:
    lim = os.getenv("CPU_LIMIT", "1")  # RSP-specific
    return int(lim)

def _have_results_check_result_map_items(
    initial: ResultMapItem, recheck:ResultMapItem, idx=int
) -> None:
    _sanity_check_result_map_items(initial, recheck, idx)
    iid = f"item #{idx} '{initial.url}'"
    for itname in ("initial", "recheck"):
        if itname == "initial":
            vitem = initial
        else:
            vitem = recheck
        for spname in ("speed_measured", "speed_reported"):
            vval=getattr(vitem,spname)
            if not vval:
                raise ValueError(
                   f"{itname} {iid} field {spname} is not set"
                )

def _sanity_check_result_map_items(
    initial: ResultMapItem, recheck:ResultMapItem, idx=int
) -> None:
        # Sanity-check the data point
        i_url = initial.url
        r_url = recheck.url
        i_sz = initial.size
        r_sz = recheck.size
        iid = f"initial item #{idx} '{i_url}'"
        if r_url != i_url:
            raise ValueError(
                f"URL for {iid}"
                f" does not match recheck item '{r_url}'"
            )
        if r_sz != i_sz:
            raise ValueError(
                f"Size {i_sz} for {iid}"
                f" does not match recheck size {r_sz}"
            )

def _get_curl() -> str:
    """Return path to ``curl`` executable."""
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("'curl' must be on path")
    return curl
    


def _t_to_spd(t_spd: str) -> float:
    """Convert speed in suffixed units to integer."""
    mega = 2 << 19
    kilo = 2 << 9
    if t_spd.endswith("M"):
        return float(t_spd[:-1]) * mega
    elif t_spd.endswith("k"):
        return float(t_spd[:-1]) * kilo
    else:
        return float(t_spd)

#### Functions for concurrency management

## Very loose typing.
type ConcurrentFunction = Callable[_ItemHelper, Any]

type _SizeCache = dict[str, int]

@dataclass
class _SpeedHelper:
    start_time: datetime.datetime
    end_time: datetime.datetime
    reported_speed: str

@dataclass
class _ItemHelper:
    index: int
    count: int
    initial_item: ResultMapItem
    recheck_item: ResultMapItem|None
    logger: logging.Logger
    curl: str
    cache: _SizeCache|None

## Concurrency limiter

async def _worker(
    sema: asyncio.Semaphore,
    fn: ConcurrentFunction,
    item: _ItemHelper
) -> None:
    async with sema:
        await fn(item)

async def _concurrency_limiter(
    fn: ConcurrentFunction,
    items: list[_ItemHelper],
    concurrency: int
) -> None:
    sema = asyncio.Semaphore(concurrency)
    workers = [_worker(sema, fn, item) for item in items]
    async with asyncio.TaskGroup() as tg:
        for wkr in workers:
            tg.create_task(wkr)

## The '_find' functions retrieve data via HTTP.

async def _find_size_for_item(
    itempair: _ItemHelper
) -> int:
    """Return size of HTTP-accessible object.
    The HEAD method is blocked, but a byte range will work."""
    size_cache = itempair.cache
    curl = itempair.curl
    idx = itempair.index
    count = itempair.count
    logger = itempair.logger
    url = itempair.initial_item.url
    if url in size_cache:
        sz = size_cache[url]
        logger.debug(f"URL '{url}' found in size cache: {sz}")
        return sz
    logger.debug(f"Finding size for item {idx+1}/{count}")
    proc = await asyncio.create_subprocess_exec(
        curl, url, "-v", "-r", "0-0", "-o" "/dev/null",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode !=0:
        raise RuntimeError(
            f"Getting object size for '{url}' failed"
            f" rc={proc.returncode},"
            f" stdout='{stdout}',"
            f" stderr='{stderr}'"
        )
    s_errl = stderr.decode().split("\n")
    # find the content-range header.
    # The total size is after the slash.
    sz = int(
        [x for x in s_errl
            if x.startswith("< content-range:")
        ][0].split("/")[-1]
    )
    size_cache[url]=sz
    itempair.initial_item.size = sz
    return sz

async def _find_speed_pair(
    itempair: _ItemHelper
) -> None:
    spdstr=f"speed for item {itempair.index+1}/{itempair.count}"
    logger=itempair.logger
    logger.debug(
        f"Finding initial {spdstr}"
    )
    await _find_speed_for_item(itempair)
    if itempair.recheck_item:
        # This is just so all the called functions have the same shape and thus
        # can be type ConcurrentFunction if we want.
        new_item = _ItemHelper(
            index=itempair.index,
            count=itempair.count,
            initial_item=itempair.recheck_item,
            recheck_item=None,
            logger=logger,
            curl=itempair.curl,
            cache=itempair.cache
        )            
        logger.debug(
            f"Finding recheck {spdstr}"
        )
        # This is sequential to allow caching to happen.  Two pulls
        # of the same URL, one after the other.  Different URLs
        # can happen concurrently.
        await _find_speed_for_item(new_item)

async def _find_speed_for_item(
    itempair: _ItemHelper
)-> None:
    """Updates result map item in place with HTTP fetch speed."""
    res = await _fetch_item(itempair)
    idx = itempair.index
    count = itempair.count
    item = itempair.initial_item
    elapsed = res.end_time - res.start_time
    speed_reported = _t_to_spd(res.reported_speed)
    speed_measured = item.size / elapsed.total_seconds()
    rm_speed_ratio = speed_reported / speed_measured
    item.start_time = res.start_time
    item.end_time = res.end_time
    item.elapsed_time = elapsed
    item.speed_reported = speed_reported
    item.speed_measured = speed_measured
    item.rm_speed_ratio = rm_speed_ratio
    itempair.logger.debug(
        f"Item {idx}/{count} "
        f"reported {speed_reported:.02f} bytes/s;"
        f" measured {speed_measured:.02f} bytes/s;"
        f" ratio (r/m) {rm_speed_ratio:.03f}"
    )

async def _fetch_item(
    itempair: _ItemHelper
) -> _SpeedHelper:
    """Get speed of transfer.

    If we're slower than 50kB/s for 30 seconds, abort.  That will break
    the run, because we won't have all the samples we expect, so no point
    in catching a bad conversion."""
    item=itempair.initial_item
    curl=itempair.curl
    proc = await asyncio.create_subprocess_exec(
        curl, item.url, "-Y", "50000", "-o", "/dev/null",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    # Reported speed ought to be slightly higher than
    # measured speed, because measured speed includes the overhead
    # of starting the process.
    #
    # We bother to track it so we can scale-test and see if at some
    # point, we have so many processes running that process thrashing
    # is taking up a significant fraction of our (wall-clock) time.
    st = _now()
    stdout, stderr = await proc.communicate()
    et = _now()
    if proc.returncode !=0:
        raise RuntimeError(
            f"Getting fetch speed for '{item.url}' failed"
            f" rc={proc.returncode},"
            f" stdout='{stdout}',"
            f" stderr='{stderr}'"
        )
    # Last field is speed (string, suffixed with binary units);
    r_spd = stderr.decode().strip().split()[-1]
    return _SpeedHelper(
        start_time=st,
        end_time=et,
        reported_speed=r_spd
    )
#### Public functions

def load_results_from_json(filename: Path) -> list[ResultMap]:
    """Read results from a persisted JSON file."""
    res: ResultSet = []
    d_obj = json.loads(filename.read_text())
    for obj in d_obj:
        res.append(RatioContainer.from_primitive(obj))
    return res

def convert_results_to_primitive(res: ResultSet) -> PrimitiveResultSet:
    """Convert results to JSON-representable form."""
    retval: PrimitiveResultSet = []
    for ratc in res:
        retval.append(ratc.to_primitive())
    return retval
    
#### Wrapper class for speed tester

class Benchmarker:
    """This object samples Butler signed URLs and determines their fetch speed.
    
    Parameters
    ----------
    butler_config
        Butler configuration, e.g. "dp2".
    butler_collections
        Butler collections to use, e.g. "dp2".
    datasetlist
        Datasets to range over during benchmarking run.
    output_dir
        Where to write output (defaults to cwd).
    concurrency
        How many fetches to run concurrently ($CPU_COUNT | 1)
    debug
        Whether to enable diagnostic output.
    recheck
        Whether to pull each URL twice, to see caching effects on speed.
    """
    
    
    def __init__(
        self,
        butler_config: str,
        butler_collections: str,
        datasetlist: DatasetList,
        output_dir: Path|None = None,
        concurrency: int|None = None,
        debug: bool = False,
        recheck: bool = True
    ) -> None:
        self._cfg = butler_config
        self._coll = butler_collections
        self._dslist = datasetlist
        self._debug = debug
        self._recheck = recheck
        if concurrency is None:
            concurrency = _cpus()
        if concurrency < 1:
            concurrency = 1
        self._concurrency = concurrency
        self._curl = _get_curl()
        self._setup_logging()
        output_dir = output_dir if output_dir else Path(".")
        self._result_path = self._setup_result_path(datadir=output_dir)
        self._size_cache: _SizeCache = {}
     
    def _setup_logging(self) -> None:
        rootlgr = logging.getLogger()
        # Get rid of root logger handlers
        for hdl in rootlgr.handlers:
            rootlgr.removeHandler(hdl)
        # Create new logger
        lgr = logging.getLogger('Weka_Benchmarker')
        level = logging.DEBUG if self._debug else logging.WARNING
        lgr.setLevel(level)
        fmt = '[%(levelname)s] %(asctime)s: %(message)s'
        for hdl in lgr.handlers:
            lgr.removeHandler(hdl)
        handler = logging.StreamHandler(stream=sys.stdout)
        lgr.addHandler(handler)
        lgr.debug("Logging initialized.")
        self._logger = lgr
        
    def _setup_result_path(self, datadir:Path) -> Path:
        """Return path to JSON file holding results."""
        nnow = _now()
        # build the name
        of_name = f"weka-gateway"
        of_name += f"-{self._cfg}-{self._coll}"
        of_name += f"-{nnow.year}.{nnow.month:02d}.{nnow.day:02d}"
        of_name += f".{nnow.hour:02d}.{nnow.minute:02d}.{nnow.second:02d}"
        of_name += f"-{os.getenv('USER')}.json"
        return datadir / of_name

    def _build_results(self) -> ResultSet:
        """From a list of parameters for a given Butler config and
        collections, create a random sample of URLs to benchmark
        (optionally rechecking for caching).
        """
        results: ResultSet = []
        butler = Butler(self._cfg, collections=self._coll)
        recheck = self._recheck
        for ds in self._dslist:
            name = ds.display_name
            dsname = ds.dataset_name
            count = ds.count
            limit = ds.limit
            if ((count <= 0) or (limit <= 0)):
                continue
            initial_result_map = ResultMap(
                display_name=name,
                dataset_name=dsname,
                count=count,
                limit=limit,
                generation=Generation.INITIAL
            )
            recheck_result_map = ResultMap(
                display_name=name,
                dataset_name=dsname,
                count=count,
                limit=limit,
                generation=Generation.RECHECK
            ) if recheck else None
            ratio_container = RatioContainer(
                display_name=name,
                dataset_name=dsname,
                count=count,
                limit=limit,          
                initial=initial_result_map,
                recheck=recheck_result_map,
            )
            results.append(ratio_container)
            self._logger.debug(
                f"Butler query of {name}[{dsname}], sample {count} from {limit}..."
            )
            q_results = butler.query_datasets(
                dsname, find_first=False, limit = limit
            )
            records = count if len(q_results) > count else len(q_results)
            urls = [str(butler.getURI(x)) for x
                    in random.sample(q_results, records)]
            r_p = '' if recheck else 'out'
            self._logger.debug(
                f"   ...sampled {records} records with{r_p} recheck"
            )
            ratio_container.count = records
            for url in urls:
                initial_result_map.result_items.append(ResultMapItem(url=url))
                if recheck:
                    recheck_result_map.result_items.append(ResultMapItem(url=url))
        return results
        
    async def _get_all_sizes(
        self, result_map: ResultMap
    ) -> ResultMap:
        """Updates result_map in place with size of each HTTP object."""
        dataset = result_map.display_name
        count=len(result_map.result_items)
        hlp: list[_ItemHelper] = []
        for idx, item in enumerate(result_map.result_items):
            hlp.append(
                _ItemHelper(
                    index=idx,
                    count=count,
                    initial_item=item,
                    recheck_item=None,
                    logger=self._logger,
                    curl=self._curl,
                    cache=self._size_cache
                )
            )
        await _concurrency_limiter(
            _find_size_for_item,
            hlp,
            self._concurrency
        )
        return result_map

    
    async def generate_results(self) -> ResultSet:
        """Get results and write them to a file.

        Returns
        -------
        ResultSet
            Results from benchmarking run.
        """
        res = self._build_results()
        for ratio_container in res:
            i_rm = ratio_container.initial
            await self._get_all_sizes(i_rm)
            r_rm: ResultMap|None = None
            if self._recheck:
                r_rm = ratio_container.recheck
                if r_rm is None:
                    raise RuntimeError(
                        "Recheck requested but no recheck result map found"
                    )
                ratio_container._prepare_recheck_items()
                ratio_container.validate_paired_resultmaps()
            count = len(i_rm.result_items)
            itempairs: list[_ItemHelper] = []
            for idx, i_item in enumerate(i_rm.result_items):
                r_item: ResultMapItem|None = None
                if self._recheck:
                    r_item = r_rm.result_items[idx]
                itempairs.append(
                    _ItemHelper(
                        index=idx,
                        count=count,
                        initial_item=i_item,
                        recheck_item=r_item,
                        logger=self._logger,
                        curl=self._curl,
                        cache=self._size_cache
                    )
                )
            await _concurrency_limiter(
                _find_speed_pair,
                itempairs,
                self._concurrency
            )
            i_rm.calculate_statistics()
            if self._recheck:
                r_rm.calculate_statistics()
                ratio_container.calculate_ratios()
        outp = convert_results_to_primitive(res)
        self._result_path.write_text(json.dumps(outp))
        return res
