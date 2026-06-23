import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Sequence

_WORKER_COMPONENT = None


class ParallelAsrModel:
    def __init__(self, component_cls: type, config: Any, num_workers: int):
        self.component_cls = component_cls
        self.config = config
        self.num_workers = max(1, int(num_workers))
        self.recommended_batch_size = self.num_workers

    def transcribe(self, batch_uttid: Sequence[str], batch_wav: Sequence[tuple[int, Any]]) -> list[dict]:
        if self.num_workers <= 1 or len(batch_uttid) <= 1:
            component = self.component_cls(self.config)
            return component.transcribe(batch_uttid, batch_wav)

        tasks = [
            (index, uttid, wav_item)
            for index, (uttid, wav_item) in enumerate(zip(batch_uttid, batch_wav))
        ]
        return _run_parallel_tasks(
            self.component_cls,
            self.config,
            self.num_workers,
            _parallel_asr_task,
            tasks,
        )


class ParallelTimestampProvider:
    def __init__(self, component_cls: type, config: Any, num_workers: int):
        self.component_cls = component_cls
        self.config = config
        self.num_workers = max(1, int(num_workers))

    def add_timestamps(self, batch_asr_result: Sequence[dict], batch_segments: Sequence[Any]) -> list[dict]:
        if self.num_workers <= 1 or len(batch_asr_result) <= 1:
            component = self.component_cls(self.config)
            results = component.add_timestamps(batch_asr_result, batch_segments)
            self.last_discarded_segments = list(getattr(component, "last_discarded_segments", []))
            return results

        tasks = [
            (index, asr_result, segment)
            for index, (asr_result, segment) in enumerate(zip(batch_asr_result, batch_segments))
        ]
        indexed_results = _run_parallel_tasks(
            self.component_cls,
            self.config,
            self.num_workers,
            _parallel_timestamp_task,
            tasks,
        )
        results = []
        discarded = []
        for item in indexed_results:
            results.extend(item["results"])
            discarded.extend(item["discarded"])
        self.last_discarded_segments = discarded
        return results


def _run_parallel_tasks(
    component_cls: type,
    config: Any,
    num_workers: int,
    worker_fn: Callable[[tuple], tuple[int, Any]],
    tasks: list[tuple],
) -> list[dict]:
    if not tasks:
        return []
    max_workers = min(num_workers, len(tasks))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_init_worker_component,
        initargs=(component_cls, config),
    ) as executor:
        indexed_results = list(executor.map(worker_fn, tasks))
    return [result for _, result in sorted(indexed_results, key=lambda item: item[0])]


def _init_worker_component(component_cls: type, config: Any) -> None:
    global _WORKER_COMPONENT
    _WORKER_COMPONENT = component_cls(config)


def _parallel_asr_task(task: tuple) -> tuple[int, dict]:
    index, uttid, wav_item = task
    [result] = _WORKER_COMPONENT.transcribe([uttid], [wav_item])
    return index, result


def _parallel_timestamp_task(task: tuple) -> tuple[int, dict]:
    index, asr_result, segment = task
    results = _WORKER_COMPONENT.add_timestamps([asr_result], [segment])
    return index, {
        "results": results,
        "discarded": list(getattr(_WORKER_COMPONENT, "last_discarded_segments", [])),
    }
