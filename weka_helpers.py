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

from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import InitVar, asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    ValidationError,
    model_validator
)
from pydantic.json_schema import SkipJsonSchema, JsonSchemaValue
from pydantic_core import core_schema
from typing import Annotated, Any, Callable, Self, Sequence, TypeVar

import httpx  # Already present in RSP

from lsst.daf.butler import Butler
from lsst.rsp import get_node, UnknownServiceError, RSPDiscovery

__all__ = [
    "Benchmarker",
    "Dataset",
    "DatasetList",
    "Generation",
    "RatioContainer",
    "RatioContainerList",
    "ResultMap",
    "ResultMapItem",
    "ResultMapItemList",
    "ResultMapList",
    "ResultStatistics",
    "convert_results_to_primitive",
    "load_results_from_json",
]

########################## JSON typing #############################
type J = dict[str,"J"] | list["J"] | str | int | float | bool | None

def strip_null(seq: Sequence) -> Sequence:
    return [x for x in seq if x is not None]

################### Data types for speed testing ###################
_T = TypeVar('_T')

class Generation(StrEnum):
    """Whether a measurement is the initial fetch or an immediate followup."""
    INITIAL = "initial"
    RECHECK = "recheck"

class Location(StrEnum):
    """Whether dataset is located at USDF or Google."""
    GOOGLE = "Google",
    USDF = "USDF"

class ResultMapItem(BaseModel):
    """Results for a single HTTP request."""

    model_config = ConfigDict(
        ser_json_temporal="seconds",
        val_temporal_unit="seconds"
    )
    
    url: str
    size: int = 0
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None
    speed_reported: float | None = None
    speed_measured: float | None = None
    node: str | None = None
    user: str | None = None
    generation: Generation | None = None
    location: Location | None = None
    count: int | None = None
    limit: int | None = None
    butler_config: str | None = None
    butler_collections: str | None = None  
    display_name: str | None = None
    dataset_name: str | None = None
    measurement_number: int | None = None
    debug: bool = False

    @property
    def elapsed_time(self) -> datetime.timedelta|None:
        """Elapsed time (wallclock) around a single measurement."""
        if not self.end_time:
            return None
        return self.end_time - self.start_time
    
    @property
    def ratio(self) -> float|None:
        """Ratio of reported (by ``curl``) speed to measured speed."""
        if not self.speed_measured:
            return None
        if self.speed_reported is None:
            return None
        return self.speed_reported / self.speed_measured

_RESULT_MAP_ITEM_SASQUATCH_SCHEMA = {
    "namespace": "lsst.benchmark",
    "type": "record",
    "name": "wekaS3Metric",
    "fields": [
        {
            "name": "url",
            "type": "string",
        },
        {
            "name": "size",
            "type": "long",
        },
        {
            "name": "start_time",
            "type": ["null", "double"],
        },
        {
            "name": "end_time",
            "type": ["null", "double"],
        },
        {
            "name": "speed_reported",
            "type": ["null", "double"],
        },
        {
            "name": "speed_measured",
            "type": ["null", "double"],
        },
        {
            "name": "node",
            "type": ["null", "string"],
        },
        {
            "name": "user",
            "type": ["null", "string"],
        },
        {
            "name": "generation",
            "type": ["null", "enum"],
            "symbols": ["initial", "recheck"],
        },
        {
            "name": "location",
            "type": ["null", "enum"],
            "symbols": ["Google", "USDF"],
        },
        {
            "name": "count",
            "type": ["null", "long"],
        },
        {
            "name": "limit",
            "type": ["null", "long"],
        },
        {
            "name": "butler_config",
            "type": ["null", "string"],
        },
        {
            "name": "butler_collections",
            "type": ["null", "string"],
        },
        {
            "name": "display_name",
            "type": ["null", "string"],
        },
        {
            "name": "dataset_name",
            "type": ["null", "string"],
        },
        {
            "name": "measurement_number",
            "type": ["null", "long"],
        },
        {
            "name": "debug",
            "type": "boolean",
            "default": "false"
        },
    ]
}

class Statistics(BaseModel):
    """Statistics for a particular list of values."""
    CV: float | None = None
    mean: float | None = None
    median: float | None = None
    pvariance: float | None = None
    pstdev: float | None = None

    @classmethod
    def from_list(cls, all_values: list[float|int|None]) -> Self:
        """Calculate statistics from a list of numbers."""
        if not all_values:
            return cls(
                CV=None,
                mean=None,
                median=None,
                pvariance=None,
                pstdev=None
            )
        nums = [ x for x in all_values if x is not None ]
        if not nums:
            return cls(
                CV=None,
                mean=None,
                median=None,
                pvariance=None,
                pstdev=None
            )            
        mean = statistics.mean(nums)
        pstdev = statistics.pstdev(nums)
        cv = pstdev / mean if mean else None
        return cls(
            mean=mean,
            median=statistics.median(nums),
            pvariance=statistics.pvariance(nums),
            pstdev=pstdev,
            CV=cv
        )

class ResultStatistics(BaseModel):
    """Container class for statistics on measurement series."""
    reported: Statistics
    measured: Statistics
    size: Statistics
    reported_measured_ratio: Statistics
    elapsed_time: Statistics

type TimeSeries = dict[datetime,float|int]
"""Mapping of instant to a numeric value at that time."""

class SpeedType(StrEnum):
    """Whether a speed is reported by ``curl`` or measured by elapsed time."""
    MEASURED = "measured"
    REPORTED = "reported"

class ResultMapItemList(UserList[_T]):
    """List of measurements with some additional methods."""
    def __init__(self, iterable: Sequence[ResultMapItem]):
        super().__init__(item for item in iterable)


    # Pydantic magic
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """We return a pydantic_core.CoreSchema that behaves in the following ways:

        * lists will be delegated to the UserList subclass.
        * Nothing else will pass validation.
        * serialization will always return just a list.
        """

        def validate_from_list(values: list[ResultMapItem|J]) -> Self:
            reval = ResultMapItemList([])
            for item in values:
                if type(item) is ResultMapItem:
                    reval.append(item)
                else:
                    reval.append(ResultMapItem.model_validate(item))
            return cls(reval)

        from_list_schema = core_schema.chain_schema(
            [
                core_schema.list_schema(),
                core_schema.no_info_plain_validator_function(validate_from_list),
            ]
        )

        
        return core_schema.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(UserList),
                    from_list_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: [ResultMapItem.model_validate(x) for x
                    in instance.data]
            ),
        )
        
    @property
    def urls(self) -> list[str]:
        """Return urls fetched for measurements."""
        return [x.url for x in self.data]

    @property
    def sizes(self) -> list[int]:
        """Return sizes associated with measurements."""
        return [x.size for x in self.data]

    @property
    def start_times(self) -> list[datetime.datetime|None]:
        """Return measurement start times."""
        return [x.start_time for x in self.data]

    @property
    def end_times(self) -> list[datetime.datetime|None]:
        """Return measurement end times."""
        return [x.end_time for x in self.data]

    @property
    def elapsed_times(self) -> list[datetime.timedelta|None]:
        """Return elapsed times of measurement."""
        return [x.elapsed_time for x in self.data]
        
    @property
    def speeds_measured(self) -> list[float|None]:
        """Return speeds as measured by elapsed time."""
        return [x.speed_measured for x in self.data]

    @property
    def speeds_reported(self) -> list[float|None]:
        """Return speeds as reported by ``curl``."""
        return [x.speed_reported for x in self.data]

    @property
    def speeds_ratio(self) -> list[float]:
        """Return ratios of reported to measured speeds."""
        return [x.ratio for x in self.data]

    @property
    def speed_deltas_reported(self) -> TimeSeries:
        """Return reported speed increases/decreases over time."""
        return self._speed_deltas(SpeedType.REPORTED)

    @property
    def speed_deltas_measured(self) -> TimeSeries:
        """Return measured speed increases/decreases over time."""
        return self._speed_deltas(SpeedType.MEASURED)

    def _speed_deltas(self, speed_type: SpeedType) -> TimeSeries:
        attr="speed_" + speed_type
        deltas = {}
        for item in self.data:
            value = getattr(item, attr)
            # It's possible that two measurements start or end
            # at the same instant.
            if item.start_time is not None:
                initial_up = deltas.get(item.start_time, 0)
                deltas[item.start_time] = initial_up + value
            if item.end_time is not None:
                initial_down = deltas.get(item.end_time, 0)
                deltas[item.end_time] = initial_down - value
        return {
            k: v for k, v in sorted(
                deltas.items(),
                key=lambda item: item[0]
            )
        }

    @property
    def events(self) -> list[datetime.datetime]:
        """Return list of times when speed changed."""
        return list(self.speed_deltas_reported.keys())

    @property
    def aggregate_speeds_reported(self) -> TimeSeries:
        """Return list of all reported speeds-at-an-instant."""
        return self._aggregate_speeds(SpeedType.REPORTED)
    
    @property    
    def aggregate_speeds_measured(self) -> TimeSeries:
        """Return list of all measured speeds-at-an-instant."""
        return self._aggregate_speeds(SpeedType.MEASURED)

    def _aggregate_speeds(self, speed_type:SpeedType) -> TimeSeries:
        total_speed = 0
        propname = "speed_deltas_" + speed_type
        delta_series = getattr(self, propname)
        speed_series:TimeSeries = {}
        for k,v in delta_series.items():
            total_speed += v
            speed_series[k] = total_speed
        return speed_series

    @property
    def aggregate_bytes(self) -> TimeSeries:
        """Return list of total bytes transferred up to each event."""
        retval: TimeSeries = {}
        events=sorted(self.data, key = lambda x: x.end_time)
        total_bytes = 0
        for ev in events:
            if ev.end_time is None:
                continue
            total_bytes += ev.size
            retval[ev.end_time] = total_bytes
        return retval

    @property
    def total_bytes(self) -> float:
        """Return total bytes transferred over the measurement series."""
        bt = list(self.aggregate_bytes.values())
        return bt[-1] if bt else 0

    @property
    def total_elapsed_time(self) -> datetime.timedelta|None:
        """Return total time taken by the measurement series."""
        if not self.events or len(self.events) < 2:
            return None
        return (self.events)[-1] - (self.events)[0]

    @property
    def average_speed(self) -> TimeSeries:
        """Return average speed as total bytes transferred over all time."""
        total_elapsed = self.total_elapsed_time
        return (
            self.total_bytes/total_elapsed.total_seconds()
        ) if total_elapsed else None
        
    @property
    def statistics(self) -> None:
        """Statistics for HTTP fetch speed, size, and times."""
        return ResultStatistics(
            reported=Statistics.from_list(self.speeds_reported),
            measured=Statistics.from_list(self.speeds_measured),
            size=Statistics.from_list(self.sizes),
            reported_measured_ratio=Statistics.from_list(self.speeds_ratio),
            elapsed_time=Statistics.from_list(
                [x.total_seconds() for x in self.elapsed_times
                if x is not None]
            )
        )

class ResultMap(BaseModel):
    """Results for a particular dataset."""

    # model_config = ConfigDict(
    #    arbitrary_types_allowed=True
    # )
    
    butler_config: str
    butler_collections: str           
    display_name: str
    dataset_name: str
    node: str | None = None
    user: str | None = None
    count: int | None = None
    limit: int | None = None
    generation: Generation | None = None
    location: Location | None = None
    debug: bool = False
    result_items: ResultMapItemList = ResultMapItemList([])

    @property
    def sizes(self) -> list[int]:
        """Return sizes associated with measurements."""
        return self.result_items.sizes

    @property
    def speeds_measured(self) -> list[float]:
        """Return speeds as measured by elapsed time."""
        return self.result_items.speeds_measured

    @property
    def speeds_reported(self) -> list[float]:
        """Return speeds as reported by ``curl``."""
        return self.result_items.speeds_reported

    @property
    def speeds_ratio(self) -> list[float]:
        """Return ratios of reported to measured speeds."""
        return self.result_items.speeds_ratio
        
    @property
    def start_times(self) -> list[datetime.datetime|None]:
        """Return measurement start times."""
        return self.result_items.start_times

    @property
    def end_times(self) -> list[datetime.datetime|None]:
        """Return measurement end times."""
        return self.result_items.end_times

    @property
    def elapsed_times(self) -> list[datetime.timedelta|None]:
        """Return elapsed times of measurement."""
        return self.result_items.elapsed_times

    @property
    def urls(self) -> list[str]:
        """Return urls fetched for measurements."""
        return self.result_items.urls

    @property
    def statistics(self) -> ResultStatistics:
        """Statistics for HTTP fetch speed, size, and times."""
        return self.result_items.statistics

    @property
    def speed_deltas_reported(self) -> TimeSeries:
        """Return reported speed increases/decreases over time."""
        return self.result_items.speed_deltas.reported

    @property
    def speed_deltas_measured(self) -> TimeSeries:
        """Return measured speed increases/decreases over time."""
        return self.result_items.speed_deltas_measured

    @property
    def events(self) -> list[datetime.datetime]:
        """Return list of times when speed changed."""
        return self.result_items.events

    @property
    def aggregate_speeds_reported(self) -> TimeSeries:
        """Return list of all reported speeds-at-an-instant."""
        return self.result_items.aggregate_speeds_reported
    
    @property    
    def aggregate_speeds_measured(self) -> TimeSeries:
        """Return list of all measured speeds-at-an-instant."""
        return self.result_items.aggregate_speeds_measured

    @property
    def aggregate_bytes(self) -> TimeSeries:
        """Return list of total bytes transferred up to each event."""
        return self.result_items.aggregate_bytes

    @property
    def total_bytes(self) -> float:
        """Return total bytes transferred over the measurement series."""
        return self.result_items.total_bytes

    @property
    def total_elapsed_time(self) -> datetime.timedelta:
        """Return total time taken by the measurement series."""
        return self.result_items.total_elapsed_time

    @property
    def average_speed(self) -> TimeSeries:
        """Return average speed as total bytes transferred over all time."""
        return self.result_items.average_speed

class ResultMapList(UserList[_T]):
    """A list of ResultMaps with some special aggregation functions."""
    def __init__(self, iterable: Sequence[ResultMap]):
        super().__init__(item for item in iterable)    

class RatioContainer(BaseModel):
    """This always holds at least one series of data.  It may hold two;
    we use this to resample a given set of URLs to investigate caching behavior."""
    butler_config: str
    butler_collections: str           
    display_name: str
    dataset_name: str
    count: int
    limit: int
    node: str
    user: str
    initial: ResultMap
    recheck: ResultMap | None = None
    ratios_measured: list[float] | None = None
    ratios_reported: list[float] | None = None

    def calculate_ratios(self) -> None:
        """Calculate ratios of initial-to-recheck speeds."""
        m_ratios: list[float] = []
        r_ratios: list[float] = []
        if not self.recheck or not self.recheck.result_items:
            return None
        self._validate_paired_resultmaps()
        for idx, i_itm in enumerate(self.initial.result_items):
            r_itm = self.recheck.result_items[idx]
            _have_results_check_result_map_items(i_itm, r_itm, idx)
            m_ratios.append(r_itm.speed_measured / i_itm.speed_measured)
            r_ratios.append(r_itm.speed_reported / i_itm.speed_reported)
        self.ratios_measured=m_ratios
        self.ratios_reported=r_ratios

        @property
        def ratio_statistics_measured(self) -> Statistics:
            Statistics.from_list(self.ratios.measured)

        @property
        def ratio_statistics_reported(self) -> Statistics:
            Statistics.from_list(self.ratios.reported)

        @property
        def measurements(self) -> list[ResultMapItem]:
            retval: list[ResultMapItem] = []
            retval.append(self.initial.result_items)
            if self.recheck:
                retval.append(self.recheck.result_items)
            return retval

    def _validate_paired_resultmaps(self) -> None:
        # Ensure that the initial and recheck maps have the same
        # number of items, and that the urls and sizes of each
        # item match.
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

class RatioContainerList(UserList[_T]):
    """A list of RatioContainers, possibly with some special properties."""
    def __init__(self, iterable: Sequence[RatioContainer]):
        super().__init__(item for item in iterable)

# Configuration object class

class Dataset(BaseModel):
    """Object to characterize source of benchmark data.

    Parameters
    ----------
    butler_config
        Butler configuration (e.g. "dp2").
    butler_collections
        Butler collections (e.g. "dp2").
    display_name
        User-friendly name for dataset collection (e.g. "big_datasets").
    dataset_name
        Internal (Butler) name for dataset collection (e.g. "run_provenance").
    count
        How many URLs to actually retrieve.
    limit
        How many URLs to get handles for from the butler.
    concurrency
        How many fetches to dispatch at once (defaults to # of CPU cores).
    recheck
        Whether to fetch each URL a second time when the first fech finishes.
    debug
        Whether to enable debugging output.
    """
    butler_config: str
    butler_collections: str
    display_name: str
    dataset_name: str
    count:int = 0
    limit:int = 10000
    location:Location | None = None
    concurrency:int | None = None
    recheck:bool = True
    debug:bool = False

class DatasetList(UserList[_T]):
    """Container for Datasets."""
    def __init__(self, iterable: Sequence[ResultMap]):
        super().__init__(item for item in iterable)
        
################### End of data types for speed testing ###################
    
#### Private utility functions

def _now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)

def _cpus() -> int:
    lim = os.getenv("CPU_LIMIT", "1")  # RSP-specific
    return int(lim)

def _get_curl() -> str:
    """Return path to ``curl`` executable."""
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("'curl' must be on path")
    return curl

def _get_user() -> str:
    return os.environ["USER"]

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

def _t_to_spd(t_spd: str) -> float:
    # Convert speed in suffixed units to integer.
    # This uses ``curl``'s conventions: SI abbreviations
    # and binary sizes (e.g. 1k=1024, not 1000).
    mega = 2 << 19
    kilo = 2 << 9
    if t_spd.endswith("M"):
        return float(t_spd[:-1]) * mega
    elif t_spd.endswith("k"):
        return float(t_spd[:-1]) * kilo
    else:
        return float(t_spd)

@dataclass
class _extracted_from_filename:
    config: str
    collections: str
    datestamp: str
    node: str
    user: str

    @classmethod
    def extract(cls, filename: Path) -> Self:
        splits = filename.name.split("-")
        lspl = len(splits)
        cfg = splits[2] if lspl > 2 else "unknown_config"
        coll = splits[3] if lspl > 3 else "unknown_collections"
        datestamp = splits[4] if lspl > 4 else "unknown_datestamp"
        node = splits[5:-1] if lspl > 6 else "unknown_node"
        user = splits[-1] if lspl > 6 else "unknown_user"
        if user.endswith(".json"):
            user = user[:-5]
        return cls(
            config=cfg,
            collections=coll,
            datestamp=datestamp,
            node=node,
            user=user
        )

def _update_ratio_container_schema(obj: J, filename: Path) -> RatioContainer:
    xt = _extracted_from_filename.extract(filename)
    updated = {
        "butler_config": xt.config,
        "butler_collections": xt.collections,
    }
    for k, v in updated.items():
        if k not in obj:
            obj[k]=v
            obj["initial"][k] = v
            if obj["recheck"]:
                obj["recheck"][k] = v
    return RatioContainer.model_validate(obj)
    

#### Functions for concurrency management

## Very loose typing.
type ConcurrentMethod = Callable[[Benchmark,_ItemPairHelper], Any]
"""A method suitable for running via semaphore and task group."""

type _SizeCache = dict[str, int]

@dataclass
class _SpeedHelper:
    start_time: datetime.datetime
    end_time: datetime.datetime
    reported_speed: str

@dataclass
class _ItemPairHelper:
    index: int
    count: int
    initial_item: ResultMapItem
    recheck_item: ResultMapItem|None
    measurement_number: int|None

#### Public functions

def load_results_from_json(filename: Path) -> RatioContainerList:
    """Read results from a persisted JSON file.
    
    Parameters
    ----------
    filename
        File to read results from.

    Returns
    -------
    RatioContainerList
        List of results for all input Datasets.
    """
    res: RatioContainerList = []
    d_obj = json.loads(filename.read_text())
    for obj in d_obj:
        try:
            pyd_obj = RatioContainer.model_validate(obj)
        except ValidationError:
            pyd_obj=_update_ratio_container_schema(obj, filename)
        res.append(pyd_obj)
    return res

def convert_results_to_primitive(res: RatioContainerList) -> dict[str,Any]:
    """Convert results to JSON-representable form."""
    retval: J = []
    for ratc in res:
        # This finesses datetime/timedelta by roundtripping it through
        # the ConfigDict temporal settings
        retval.append(json.loads(ratc.model_dump_json()))
    return retval
    
#### Speed Tester class

class Benchmarker:
    """This object samples Butler signed URLs and determines their fetch speed.
    
    Parameters
    ----------
    dataset_list
        Datasets to range over during benchmarking run.
    output_dir
        Path to write output to (defaults to cwd).
    concurrency
        Number of fetches to do concurrently (defaults to $CPU_LIMIT or 1).
    debug
        Whether to emit debugging output (defaults to False).
    sasquatch_url
        Sasquatch URL to receive data, or None (defaults to None).
    """
        
    def __init__(
        self,
        dataset_list: DatasetList,
        output_dir: Path|None = None,
        concurrency: int|None = None,
        debug: bool = False,
        send_to_sasquatch: bool = False,
    ) -> None:
        self._dslist = dataset_list
        self._debug = debug
        if concurrency is None:
            concurrency = _cpus()
        if concurrency < 1:
            concurrency = 1
        self._cfg, self._coll = self._get_cfg_and_coll()
        self._concurrency = concurrency
        self._debug = debug
        self._sasquatch_client: httpx.AsyncClient | None = None
        self._send_to_sasquatch = send_to_sasquatch
        self._node = get_node()
        self._curl = _get_curl()
        self._setup_logging()
        output_dir = output_dir if output_dir else Path(".")
        self._result_path = self._setup_result_path(datadir=output_dir)
        self._size_cache: _SizeCache = {}

    def _get_cfg_and_coll(self) -> tuple[str]:
        cfg = "+".join(list(set([x.butler_config for x in self._dslist])))
        coll = "+".join(list(set([x.butler_collections 
                                  for x in self._dslist])))
        return (cfg,coll)
    
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
        of_name += f"-{self._node}"
        of_name += f"-{_get_user()}.json"
        return datadir / of_name

    async def _get_sasquatch_client(self) -> httpx.AsyncClient|None:
        """Acquire a sasquatch client, if we can."""
        token=RSPDiscovery.get_token()
        wrn="; will not report to Sasquatch."
        if not token:
            self._logger.warning(
                f"Could not acquire access token{wrn}"
            )
            return None
        # It doesn't look like lsst.rsp gives us a way to get internal services.
        # We also don't have rubin.repertoire available in the environment.
        inst_url=os.getenv("EXTERNAL_INSTANCE_URL", None)
        if not inst_url:
            self._logger.warning(
                f"Could not determine RSP instance URL{wrn}"
            )
            return None
        client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            follow_redirects=True,
        )
        resp = await client.get(f"{inst_url}/auth/api/v1/token-info")
        resp.raise_for_status()
        obj = resp.json()
        scopes = obj.get("scopes",[])
        if "write:sasquatch" not in scopes:
            self._logger.warning(
                f"'write:sasquatch' not in scopes {scopes}"
            )
            return None
        base_url = f"{inst_url}/sasquatch-rest-proxy"
        client.base_url = base_url
        return client

    async def _configure_topic(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/v3/clusters")
        resp.raise_for_status()
        obj = resp.json()
        cluster_id = obj["data"][0]["cluster_id"]
        topic_config = {
            "topic_name": "lsst.benchmark.wekaS3Metric",
            "partitions_count": 1,
            "replication_factor": 3
        }
        resp = await client.post(
            f"/v3/clusters/{cluster_id}/topics",
            json=topic_config
        )
        resp.raise_for_status()

    def _assemble_measurements(self, res: RatioContainerList) -> list[J]:
        """Create a list of JSON representations of all ResultMapItems."""
        meas: list[J] = []
        for ratc in res:
            meas.append([ x.model_dump_json() for x in res.measurements ])
        # Chronograf expects a column named "timestamp".
        for m in meas:
            m["timestamp"] = m["start_time"]
        return meas
    
    async def _send_results_to_sasquatch(
        self,
        res: RatioContainerList
    ) -> None:
        """Send results to Sasquatch, if it's configured."""
        if len(res) < 1:
            return
        if not self._send_to_sasquatch:
            return
        client = await self._get_sasquatch_client()
        if not client:
            return
        await self._configure_topic(client)
        meas = self._assemble_measurements()
        # Put together the Avro schema definition and the measurements.
        records = [ { "value": m } for m in meas ]
        payload = {
            "value_schema": json.dumps(
                _RESULT_MAP_ITEM_SASQUATCH_SCHEMA
            ),
            "records": records
        }
        headers = {
            "Content-Type": "application/vnd.kafka.avro.v2+json",
            "Accept": "application/vnd.kafka.v2+json",
        }
        resp = await client.post(
            "/topics/lsst.benchmark.wekaS3Metric",
            json=payload,
            headers=headers
        )
        await client.aclose()
        resp.raise_for_status()
        self._logger.debug(
            f"Measurements sent to Sasquatch at {client.base_url}"
        )

    ## Concurrency limiter

    async def _worker(
        self,
        sema: asyncio.Semaphore,
        meth: ConcurrentMethod,
        item: _ItemPairHelper
    ) -> None:
        async with sema:
            await meth(item)

    async def _concurrency_limiter(
        self,
        meth: ConcurrentMethod,
        items: list[_ItemPairHelper],
    ) -> None:
        sema = asyncio.Semaphore(self._concurrency)
        workers = [self._worker(sema, meth, item) for item in items]
        async with asyncio.TaskGroup() as tg:
            for wkr in workers:
                tg.create_task(wkr)

    ## The '_find' methods retrieve data via HTTP.

    async def _find_size_for_item(
        self,
        itempair: _ItemPairHelper
    ) -> int:
        """Return size of HTTP-accessible object.
        The HEAD method is blocked, but a byte range will work."""
        size_cache = self._size_cache
        curl = self._curl
        idx = itempair.index
        count = itempair.count
        measurement_number = itempair.measurement_number
        url = itempair.initial_item.url
        if url in size_cache:
            sz = size_cache[url]
            self.logger.debug(f"URL '{url}' found in size cache: {sz}")
            return sz
        self._logger.debug(f"Finding size for item {idx+1}/{count}")
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
        self._logger.debug(f"...{idx+1}/{count}: {sz}B")
        return sz

    async def _find_speed_pair(
        self,
        itempair: _ItemPairHelper,
    ) -> None:
        itemstr=f"item {itempair.index+1}/{itempair.count}"
        spdstr=f"speed for {itemstr}"
        self._logger.debug(
            f"Finding initial {spdstr}"
        )
        await self._determine_speed_for_item(
            itempair.initial_item,
            itemstr,
            itempair.measurement_number,
        )
        if itempair.recheck_item:
            self._logger.debug(
                f"Finding recheck {spdstr}"
            )
            # This is sequential to allow caching to happen.  Two pulls
            # of the same URL, one after the other.  Different URLs
            # can happen concurrently.
            await self._determine_speed_for_item(
                itempair.recheck_item,
                itemstr,
                itempair.measurement_number,
            )

    ## Further helper methods for getting speed.

    async def _determine_speed_for_item(
        self,
        item: ResultMapItem,
        itemstr: str,
        measurement_number: int
    )-> None:
        """Updates result map item in place with HTTP fetch speed.
        
        Inserts reported speed, and calculates measured speed and
        ratio of reported to measured speeds.
        """
        res = await self._fetch_item(item.url)
        elapsed = res.end_time - res.start_time
        speed_reported = _t_to_spd(res.reported_speed)
        speed_measured = item.size / elapsed.total_seconds()
        item.start_time = res.start_time
        item.end_time = res.end_time
        item.speed_reported = speed_reported
        item.speed_measured = speed_measured
        item.measurement_number = measurement_number
        self._logger.debug(
            f"{itemstr}:"
            f" reported {speed_reported:.02f} bytes/s;"
            f" measured {speed_measured:.02f} bytes/s;"
            f" ratio (r/m) {speed_reported/speed_measured:.03f}"
        )

    async def _fetch_item(
        self,
        url: str
    ) -> _SpeedHelper:
        """Get speed of transfer.

        If we're slower than 50kB/s for 30 seconds, abort.  That will break
        the run, because we won't have all the samples we expect, so no point
        in catching a bad conversion."""
        curl=self._curl
        # If we're slower than 50kB/s for 30 seconds, abort.
        # That will break the run, since we will have None for some of
        # our sample values (the tasks will throw exceptions and
        # not update their speed fields).
        proc = await asyncio.create_subprocess_exec(
            curl, url, "-Y", "50000", "-o", "/dev/null",
            stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
        )
        # Reported speed ought to be slightly higher than
        # measured speed, because measured speed includes the overhead
        # of the curl process outside its fetch loop too.
        #
        # However, measured speed is much more granular, so the rounding
        # inherent in reported speed might make it slightly slower.
        # If reported speed varies outside the window of "slightly slower"
        # to "somewhat faster" then....
        #
        # ...that's why we track it: we can scale-test and see if at some
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
        # Last field is speed (string, suffixed with binary units).
        r_spd = stderr.decode().strip().split()[-1]
        return _SpeedHelper(
            start_time=st,
            end_time=et,
            reported_speed=r_spd
        )
    
    def _build_results(self) -> RatioContainerList:
        """From a list of parameters for a given Butler config and
        collections, create a random sample of URLs to benchmark
        (optionally rechecking for caching).
        """
        results: RatioContainerList = []
        for ds in self._dslist:
            butler = Butler(
                ds.butler_config,
                collections=ds.butler_collections
            )
            name = ds.display_name
            dsname = ds.dataset_name
            count = ds.count
            limit = ds.limit
            recheck = ds.recheck
            cfg = ds.butler_config
            coll = ds.butler_collections
            user = _get_user()
            node = self._node
            location = ds.location
            debug = self._debug
            if ((count <= 0) or (limit <= 0)):
                continue
            initial_result_map = ResultMap(
                butler_config=cfg,
                butler_collections=coll,
                display_name=name,
                dataset_name=dsname,
                count=count,
                limit=limit,
                generation=Generation.INITIAL,
                user=user,
                node=node,
                location=location,
            )
            recheck_result_map = ResultMap(
                butler_config=ds.butler_config,
                butler_collections=ds.butler_collections,
                display_name=ds.display_name,
                dataset_name=ds.dataset_name,
                count=count,
                limit=limit,
                generation=Generation.RECHECK,
                user=user,
                node=node,
                location=location,
                debug=debug,
            ) if ds.recheck else None
            ratio_container = RatioContainer(
                butler_config=cfg,
                butler_collections=coll,
                display_name=name,
                dataset_name=dsname,
                node=self._node,
                count=count,
                limit=limit,
                user=user,
                initial=initial_result_map,
                recheck=recheck_result_map,
                location=location,
                debug=debug,
            )
            results.append(ratio_container)
            self._logger.debug(
                f"Butler query of {name}[{dsname}], "
                f"sample {count} from {limit}..."
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
                initial_result_map.result_items.append(
                    ResultMapItem(
                        url=url,
                        node=node,
                        user=user,
                        count=count,
                        limit=limit,
                        butler_config=cfg,
                        butler_collections=coll,
                        display_name=name,
                        dataset_name=dsname,
                        generation=Generation.INITIAL,
                        location=location,
                        debug=debug,
                    ))
                if recheck:
                    recheck_result_map.result_items.append(
                        ResultMapItem(
                            url=url,
                            node=node,
                            user=user,
                            count=count,
                            limit=limit,
                            butler_config=cfg,
                            butler_collections=coll,
                            display_name=name,
                            dataset_name=dsname,
                            generation=Generation.RECHECK,
                            location=location,
                            debug=debug,
                        ))
        return results
        
    async def _get_all_sizes(
        self, result_map: ResultMap
    ) -> ResultMap:
        """Updates result_map in place with size of each HTTP object."""
        dataset = result_map.display_name
        count=len(result_map.result_items)
        hlp: list[_ItemPairHelper] = []
        for idx, item in enumerate(result_map.result_items):
            hlp.append(
                _ItemPairHelper(
                    index=idx,
                    count=count,
                    initial_item=item,
                    recheck_item=None,
                    measurement_number=idx+1
                )
            )
        await self._concurrency_limiter(
            self._find_size_for_item,
            hlp,
        )
        return result_map

    async def generate_results(self) -> RatioContainerList:
        """Get results, write them to a file, and return them.

        Returns
        -------
        RatioContainerList
            Results from benchmarking run.
        """
        res = self._build_results()
        for ratio_container in res:
            i_rm = ratio_container.initial
            r_rm = ratio_container.recheck
            await self._get_all_sizes(i_rm)
            if r_rm:        
                ratio_container._prepare_recheck_items()
                ratio_container._validate_paired_resultmaps()
            count = len(i_rm.result_items)
            itempairs: list[_ItemPairHelper] = []
            for idx, i_item in enumerate(i_rm.result_items):
                r_item = (ratio_container.recheck.result_items[idx]
                          if ratio_container.recheck else None)
                itempairs.append(
                    _ItemPairHelper(
                        index=idx,
                        count=count,
                        initial_item=i_item,
                        recheck_item=r_item,
                        measurement_number=idx+1
                    )
                )
            await self._concurrency_limiter(
                self._find_speed_pair,
                itempairs,
            )
            ratio_container.initial.statistics
            if ratio_container.recheck:
                ratio_container.recheck.statistics
                ratio_container.calculate_ratios()
        outp = convert_results_to_primitive(res)
        self._result_path.write_text(json.dumps(outp))
        await self._send_results_to_sasquatch(res)
        return res
