from threading import Lock
from typing import Optional

from modules.person.Person import Person, TrackingStatus

from modules.utils.HotReloadMethods import HotReloadMethods

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

    @property
    def available(self) -> list[int]:
        with self._lock:
            return sorted(self._available)

class PersonManager:
    def __init__(self, max_persons: int) -> None:
        self._persons: dict[int, Person] = {}
        self._id_pool = PersonIdPool(max_persons)
        self._lock = Lock()

        hot_reload = HotReloadMethods(self.__class__, True)

    def add_person(self, person: Person) -> Optional[int]:
        with self._lock:
            try:
                person_id = self._id_pool.acquire()
            except Exception as e:
                print(f"PersonManager: No more IDs available: {e}")
                return None
            person.id = person_id
            person.status = TrackingStatus.NEW
            self._persons[person_id] = person
            return person_id

    def get_person(self, person_id: int) -> Optional[Person]:
        with self._lock:
            return self._persons.get(person_id, None)

    def set_person(self, person: Person) -> None:
        with self._lock:
            if person.id in self._persons:
                self._persons[person.id] = person
            else:
                print(f"PersonManager: Attempted to set non-existent person with ID {person.id}. Adding as new person.")

    def remove_person(self, person_id: int) -> None:
        with self._lock:
            person: Person | None = self._persons.pop(person_id, None)
            # person.status = TrackingStatus.REMOVED
            if person is not None:
                self._id_pool.release(person_id)
            else:
                print(f"PersonManager: Attempted to remove non-existent person with ID {person_id}.")

    def all_persons(self) -> list[Person]:
        with self._lock:
            return list(self._persons.values())

    def get_person_by_cam_and_tracklet(self, cam_id: int, tracklet_id: int) -> Optional[Person]:
        with self._lock:
            for person in self._persons.values():
                if person.cam_id == cam_id and person.tracklet and person.tracklet.id == tracklet_id:
                    return person
            return None

    def replace_person(self, old_person: Person, new_person: Person) -> None:
        """
        Replace an existing person in the manager with a new person object.

        The new person will:
        - Take the id and start_time from the old person.
        - Have its status set to TRACKED if its current status is NEW.
        - Replace the old person in the manager's dictionary.

        Args:
            old_person (Person): The person currently in the manager to be replaced.
            new_person (Person): The new person object to insert, inheriting id and start_time from old_person.
        """
        if self.get_person(old_person.id) is None:
            print(f"PersonManager: Attempted to replace non-existent person with ID {old_person.id}.")
            return

        with self._lock:
            # Transfer id and start_time
            new_person.id = old_person.id
            new_person.start_time = old_person.start_time

            # Update status if needed
            if new_person.status == TrackingStatus.NEW:
                new_person.status = TrackingStatus.TRACKED

            if new_person.status == TrackingStatus.LOST:
                new_person.last_time = old_person.last_time

            if new_person.status == TrackingStatus.REMOVED:
                print(f"PersonManager: Attempted to replace person with ID {new_person.id} with status REMOVED. This should not happen.")
                # return

            # Replace in the dict
            self._persons[old_person.id] = new_person

    def merge_persons(self, keep: Person, remove: Person) -> int:
        """
        Merge two Person objects into a single entry in the manager.

        The resulting person will:
        - Use the id and start_time of the older person (the one with the earlier start_time).
        - Use all other attributes from the 'keep' person.
        - Remove both original persons from the manager and release the ID of the newer person.
        - Add the merged person back to the manager with the merged id.

        Args:
            keep (Person): The person whose data (except id and start_time) will be kept.
            remove (Person): The person whose id and start_time may be used if older.

        Returns:
            int: The id of the person that was removed and released, or -1 if the merge was not successful.
        """

        with self._lock:
            # Check for invalid IDs
            if keep.id in (-1, None):
                print(f"PersonManager: Cannot merge persons with uninitialized id (keep.id={keep.id}, remove.id={remove.id})")
                return -1

            if keep.id == remove.id:
                print(f"PersonManager: Attempted to merge the same person {keep.id}.")
                return -1

            # Determine which person is oldest
            if keep.age >= remove.age:
                oldest, newest = keep, remove
            else:
                oldest, newest = remove, keep

            # Save the id and start_time of the oldest
            merged_id: int = oldest.id
            other_id: int = newest.id
            merged_start_time: float = oldest.start_time

            # Use all other data from the newest (the 'keep' person)
            merged_person = Person(
                id=merged_id,
                cam_id=keep.cam_id,
                tracklet=keep.tracklet,
                time_stamp=keep.time_stamp
            )

            # Set the id and start_time from the oldest
            merged_person.start_time = merged_start_time

            # Copy all relevant fields from 'keep'
            merged_person.status = keep.status
            if keep.status == TrackingStatus.NEW:
                merged_person.status = TrackingStatus.TRACKED
            merged_person.last_time = keep.last_time
            merged_person.local_angle = keep.local_angle
            merged_person.world_angle = keep.world_angle
            merged_person.overlap = keep.overlap
            merged_person.img = keep.img
            merged_person.pose_roi = keep.pose_roi
            merged_person.pose = keep.pose
            merged_person.pose_angles = keep.pose_angles

            # Remove both old persons from the manager
            self._persons.pop(merged_id, None)
            if other_id != -1:
                self._persons.pop(other_id, None)
                if other_id != merged_id:
                    self._id_pool.release(other_id)

            # Add the merged person with merged_id
            self._persons[merged_id] = merged_person

            return other_id

