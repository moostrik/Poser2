from threading import Lock
from typing import Optional

from modules.person.Person import Person

class PersonIdPool:
    def __init__(self, max_size: int) -> None:
        self._available: set[int] = set(range(max_size))
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._available:
                raise Exception("No more IDs available")
            min_id: int = min(self._available)
            self._available.remove(min_id)
            return min_id

    def release(self, obj: int) -> None:
        with self._lock:
            if obj in self._available:
                raise Exception(f"ID {obj} is not currently in use and cannot be released. in use: {self._available}")
            self._available.add(obj)

    def size(self) -> int:
        return len(self._available)

    def is_available(self, obj: int) -> bool:
        with self._lock:
            return obj in self._available

class PersonManager:
    def __init__(self, max_persons: int) -> None:
        self._persons: dict[int, Person] = {}
        self._id_pool = PersonIdPool(max_persons)
        self._lock = Lock()

    def add_person(self, person: Person) -> Optional[int]:
        with self._lock:
            try:
                person_id = self._id_pool.acquire()
            except Exception as e:
                print(f"PersonManager: No more IDs available: {e}")
                return None
            person.id = person_id
            self._persons[person_id] = person
            return person_id

    def get_person(self, person_id: int) -> Optional[Person]:
        with self._lock:
            return self._persons.get(person_id, None)

    def get_person_by_cam_and_tracklet(self, cam_id: int, tracklet_id: int) -> Optional[Person]:
        with self._lock:
            for person in self._persons.values():
                if person.cam_id == cam_id and person.tracklet.id == tracklet_id:
                    return person
            return None

    def set_person(self, person: Person) -> Optional[int]:
        with self._lock:
            if person.id in self._persons:
                self._persons[person.id] = person
                return person.id
            else:
                return None

    def remove_person(self, person_id: int) -> None:
        with self._lock:
            person: Person | None = self._persons.pop(person_id, None)
            if person is not None:
                self._id_pool.release(person_id)

    def all_persons(self) -> list[Person]:
        with self._lock:
            return list(self._persons.values())