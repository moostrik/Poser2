from typing import Type, TypeVar, List

T = TypeVar('T')

class ObjectPool:
    def __init__(self, cls: Type[T], size: int, *args, **kwargs) -> None:
        self._cls = cls
        self._size = size
        self._args = args
        self._kwargs = kwargs
        self._pool: List[T] = [self._create_instance() for _ in range(size)] # type: ignore
        self._available: List[T] = self._pool.copy() # type: ignore

    def _create_instance(self) -> T: # type: ignore
        return self._cls(*self._args, **self._kwargs) # type: ignore

    def acquire(self) -> T: # type: ignore
        if not self._available:
            raise Exception("No available instances in the pool")
        return self._available.pop()

    def release(self, instance: T) -> None: # type: ignore
        if instance not in self._pool:
            raise Exception("Instance does not belong to the pool")
        self._available.append(instance)

    def size(self) -> int:
        return len(self._pool)

    def available(self) -> int:
        return len(self._available)

    def get_all_objects(self) -> List[T]: # type: ignore
        return self._pool