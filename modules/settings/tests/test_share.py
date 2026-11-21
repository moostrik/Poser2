"""Sharing: a parent's field shared down to its children and a child's write shared up."""

import unittest

from modules.settings import Field, Widget, BaseSettings, Group

from .fixtures import MinimalSettings, CoreSettings, PlayerSettings, ParentWithChild


class UpwardChild(BaseSettings):
    status = Field(0)


class UpwardParent(BaseSettings):
    status = Field(0)
    child = Group(UpwardChild, share=[status])


class TestGroupShare(unittest.TestCase):
    """Tests for share=[fps, color] — bidirectional sharing between parent and children."""

    def test_shared_values_propagated(self):
        s = ParentWithChild()
        for core in s.cores:
            self.assertEqual(core.fps, 30.0)
            self.assertEqual(core.color, True)

    def test_shared_from_kwargs(self):
        s = ParentWithChild(fps=60.0, color=False)
        for core in s.cores:
            self.assertEqual(core.fps, 60.0)
            self.assertEqual(core.color, False)

    def test_shared_excluded_from_child_serialization(self):
        s = ParentWithChild()
        d = s.to_dict()
        for i in range(3):
            core_dict = d[f"core_{i}"]
            self.assertNotIn("fps", core_dict)
            self.assertNotIn("color", core_dict)
            self.assertIn("exposure", core_dict)

    def test_shared_present_on_parent_serialization(self):
        s = ParentWithChild()
        d = s.to_dict()
        self.assertIn("fps", d)
        self.assertIn("color", d)

    def test_non_shared_fields_not_affected(self):
        s = ParentWithChild()
        self.assertEqual(s.core_0.exposure, 1000)

    def test_share_re_propagated_on_update_from_dict(self):
        s = ParentWithChild()
        s.update_from_dict({"fps": 90.0})
        for core in s.cores:
            self.assertEqual(core.fps, 90.0)

    def test_runtime_assignment_propagates_downstream(self):
        class SharedValue(BaseSettings):
            value = Field(0)

        class LiveParent(BaseSettings):
            level = Field(1)
            child = Group(SharedValue, share=[level.as_('value')])

        s = LiveParent()
        s.level = 42
        self.assertEqual(s.child.value, 42)

    def test_runtime_assignment_propagates_to_grandchildren(self):
        class SharedLeaf(BaseSettings):
            value = Field(0)

        class SharedBranch(BaseSettings):
            value = Field(0)
            leaf = Group(SharedLeaf, share=[value])

        class LiveParent(BaseSettings):
            level = Field(1)
            child = Group(SharedBranch, share=[level.as_('value')])

        s = LiveParent()
        s.level = 7
        self.assertEqual(s.child.value, 7)
        self.assertEqual(s.child.leaf.value, 7)

    def test_child_side_write_propagates_upward(self):
        class SharedValue(BaseSettings):
            value = Field(0)

        class LiveParent(BaseSettings):
            level = Field(1)
            child = Group(SharedValue, share=[level.as_('value')])

        s = LiveParent()
        s.child.value = 9
        self.assertEqual(s.level, 9)

    def test_a_shared_field_says_so(self):
        class SharedValue(BaseSettings):
            value = Field(0)

        class LiveParent(BaseSettings):
            level = Field(1)
            child = Group(SharedValue, share=[level.as_('value')])

        s = LiveParent()
        self.assertTrue(s.child.is_shared('value'))

    def test_round_trip(self):
        s1 = ParentWithChild(fps=45.0, color=False)
        s1.core_0.exposure = 2000
        s1.core_2.exposure = 500
        data = s1.to_dict()

        s2 = ParentWithChild()
        s2.update_from_dict(data)
        self.assertEqual(s2.fps, 45.0)
        self.assertEqual(s2.core_0.fps, 45.0)
        self.assertEqual(s2.core_0.exposure, 2000)
        self.assertEqual(s2.core_2.exposure, 500)


class TestPropagatSharedAfterInitialize(unittest.TestCase):
    """_propagate_shared must not raise when called after initialize().

    Regression test for the bug where update_from_dict() called
    _propagate_shared() which tried to set INIT fields on an already-
    initialized child, raising AttributeError.
    """

    def test_update_from_dict_after_initialize_does_not_raise(self):
        """Calling update_from_dict after initialize() must not raise."""
        s = ParentWithChild()
        s.initialize()
        # Should not raise even though fps/color are shared INIT fields
        s.update_from_dict({"core_0": {"exposure": 500}})
        self.assertEqual(s.core_0.exposure, 500)

    def test_init_shared_fields_not_updated_after_initialize(self):
        """Shared INIT fields in children must stay frozen after initialize()."""
        s = ParentWithChild(fps=30.0)
        s.initialize()
        # Even if the parent dict contains a new fps, children must keep their frozen value
        s.update_from_dict({"fps": 60.0, "core_0": {"exposure": 999}})
        # fps on parent is also INIT → skipped by update_from_dict
        self.assertEqual(s.fps, 30.0)
        self.assertEqual(s.core_0.fps, 30.0)

    def test_non_init_shared_fields_still_propagate_after_initialize(self):
        """Non-INIT shared fields must still propagate after initialize()."""
        class LiveParent(BaseSettings):
            level = Field(1)  # WRITE — not INIT
            child = Group(MinimalSettings, share=[level.as_('value')])

        s = LiveParent()
        s.initialize()
        s.update_from_dict({"level": 42})
        # level is WRITE so update_from_dict updates it, then propagates to child
        self.assertEqual(s.level, 42)
        self.assertEqual(s.child.value, 42)


class TestGroupUpwardShare(unittest.TestCase):
    """Tests for child → parent propagation via share=[...]."""

    def test_child_write_propagates_to_parent(self):
        s = UpwardParent()
        s.child.status = 42
        self.assertEqual(s.status, 42)

    def test_parent_write_propagates_to_child(self):
        s = UpwardParent()
        s.status = 99
        self.assertEqual(s.child.status, 99)

    def test_initial_value_from_parent(self):
        s = UpwardParent()
        self.assertEqual(s.status, 0)
        self.assertEqual(s.child.status, 0)

    def test_upward_with_alias(self):
        class AliasChild(BaseSettings):
            output = Field(0)

        class AliasParent(BaseSettings):
            result = Field(0)
            child = Group(AliasChild, share=[result.as_('output')])

        s = AliasParent()
        s.child.output = 7
        self.assertEqual(s.result, 7)

    def test_shared_excluded_from_child_serialization(self):
        s = UpwardParent()
        s.child.status = 5
        d = s.to_dict()
        self.assertIn("status", d)
        self.assertNotIn("status", d["child"])

    def test_shared_present_on_parent_serialization(self):
        s = UpwardParent()
        s.child.status = 5
        d = s.to_dict()
        self.assertEqual(d["status"], 5)

    def test_deserialization_propagates_to_child(self):
        s = UpwardParent()
        s.update_from_dict({"status": 99})
        self.assertEqual(s.child.status, 99)

    def test_callback_on_parent(self):
        s = UpwardParent()
        captured = []
        s.bind(UpwardParent.status, lambda v: captured.append(v))
        s.child.status = 10
        self.assertEqual(captured, [10])

    def test_callback_on_child(self):
        s = UpwardParent()
        captured = []
        s.child.bind(UpwardChild.status, lambda v: captured.append(v))
        s.child.status = 10
        self.assertEqual(captured, [10])

    def test_round_trip(self):
        s1 = UpwardParent()
        s1.child.status = 42
        data = s1.to_dict()
        s2 = UpwardParent()
        s2.update_from_dict(data)
        self.assertEqual(s2.child.status, 42)
        self.assertEqual(s2.status, 42)

    def test_share_access_mismatch_raises(self):
        """INIT parent + non-INIT child must raise (asymmetric access)."""
        with self.assertRaises(TypeError) as ctx:
            class ChildNonInit(BaseSettings):
                val = Field(0)  # READWRITE

            class ParentInit(BaseSettings):
                val = Field(0, access=Field.INIT)
                child = Group(ChildNonInit, share=[val])

            ParentInit()
        self.assertIn("access mismatch", str(ctx.exception))

    def test_no_infinite_loop(self):
        s = UpwardParent()
        s.status = 5
        self.assertEqual(s.status, 5)
        self.assertEqual(s.child.status, 5)
        s.child.status = 7
        self.assertEqual(s.status, 7)
        self.assertEqual(s.child.status, 7)

    def test_fan_out_to_sibling_children(self):
        """Child write fans out via parent to sibling children."""
        class SiblingChild(BaseSettings):
            level = Field(0)

        class FanOutParent(BaseSettings):
            level = Field(0)
            writer = Group(SiblingChild, share=[level])
            reader = Group(SiblingChild, share=[level])

        s = FanOutParent()
        s.writer.level = 50
        self.assertEqual(s.level, 50)
        self.assertEqual(s.reader.level, 50)

    def test_callback_fires_once_per_direction(self):
        s = UpwardParent()
        parent_calls = []
        child_calls = []
        s.bind(UpwardParent.status, lambda v: parent_calls.append(v))
        s.child.bind(UpwardChild.status, lambda v: child_calls.append(v))
        s.child.status = 33
        self.assertEqual(child_calls, [33])
        self.assertEqual(parent_calls, [33])

    def test_upward_propagation_through_grandparent(self):
        """Grandchild write propagates upward through child to grandparent."""
        class GrandChild(BaseSettings):
            val = Field(0)

        class Middle(BaseSettings):
            val = Field(0)
            gc = Group(GrandChild, share=[val])

        class Top(BaseSettings):
            val = Field(0)
            mid = Group(Middle, share=[val])

        s = Top()
        s.mid.gc.val = 77
        self.assertEqual(s.mid.val, 77)
        self.assertEqual(s.val, 77)

    def test_downward_propagation_through_grandchild(self):
        """Grandparent write propagates downward through child to grandchild."""
        class GrandChild(BaseSettings):
            val = Field(0)

        class Middle(BaseSettings):
            val = Field(0)
            gc = Group(GrandChild, share=[val])

        class Top(BaseSettings):
            val = Field(0)
            mid = Group(Middle, share=[val])

        s = Top()
        s.val = 88
        self.assertEqual(s.mid.val, 88)
        self.assertEqual(s.mid.gc.val, 88)

    def test_fire_propagates_downward(self):
        """Firing a shared button on parent fires callbacks on child."""
        class BtnChild(BaseSettings):
            go = Field(False, widget=Widget.button)

        class BtnParent(BaseSettings):
            go = Field(False, widget=Widget.button)
            child = Group(BtnChild, share=[go])

        s = BtnParent()
        child_calls = []
        s.child.bind(BtnChild.go, lambda v: child_calls.append(v))
        BtnParent.go.fire(s)
        self.assertEqual(child_calls, [True])

    def test_fire_propagates_upward(self):
        """Firing a shared button on child fires callbacks on parent."""
        class BtnChild(BaseSettings):
            go = Field(False, widget=Widget.button)

        class BtnParent(BaseSettings):
            go = Field(False, widget=Widget.button)
            child = Group(BtnChild, share=[go])

        s = BtnParent()
        parent_calls = []
        s.bind(BtnParent.go, lambda v: parent_calls.append(v))
        BtnChild.go.fire(s.child)
        self.assertEqual(parent_calls, [True])

    def test_fire_no_infinite_recursion(self):
        """Shared button fire does not loop between parent and child."""
        class BtnChild(BaseSettings):
            go = Field(False, widget=Widget.button)

        class BtnParent(BaseSettings):
            go = Field(False, widget=Widget.button)
            a = Group(BtnChild, share=[go])
            b = Group(BtnChild, share=[go])

        s = BtnParent()
        calls = []
        s.bind(BtnParent.go, lambda v: calls.append('parent'))
        s.a.bind(BtnChild.go, lambda v: calls.append('a'))
        s.b.bind(BtnChild.go, lambda v: calls.append('b'))
        BtnParent.go.fire(s)  # must not blow the stack
        self.assertIn('parent', calls)
        self.assertIn('a', calls)
        self.assertIn('b', calls)

    def test_fire_fan_out_from_child(self):
        """Firing on one child fans out via parent to sibling."""
        class BtnChild(BaseSettings):
            go = Field(False, widget=Widget.button)

        class BtnParent(BaseSettings):
            go = Field(False, widget=Widget.button)
            writer = Group(BtnChild, share=[go])
            reader = Group(BtnChild, share=[go])

        s = BtnParent()
        parent_calls, reader_calls = [], []
        s.bind(BtnParent.go, lambda v: parent_calls.append(v))
        s.reader.bind(BtnChild.go, lambda v: reader_calls.append(v))
        BtnChild.go.fire(s.writer)
        self.assertEqual(parent_calls, [True])
        self.assertEqual(reader_calls, [True])


class _PairChild(BaseSettings):
    a = Field(0)
    b = Field(0)


class _PairParent(BaseSettings):
    a = Field(0)
    b = Field(0)
    child = Group(_PairChild, share=[a, b])


class TestShareDirection(unittest.TestCase):
    """A share's direction belongs to the write it is part of, not to the instance."""

    def test_a_child_write_during_a_push_down_reaches_the_parent(self):
        parent = _PairParent()
        # while the parent pushes `a` down into the child, the child's callback writes its `b`
        parent.child.bind(_PairChild.a, lambda value: setattr(parent.child, "b", value * 10))
        parent.a = 3
        self.assertEqual(parent.child.b, 30)
        self.assertEqual(parent.b, 30)

    def test_a_button_pressed_on_a_child_reaches_the_parent(self):
        class Child(BaseSettings):
            go = Field(False, widget=Widget.button)

        class Parent(BaseSettings):
            go = Field(False, widget=Widget.button)
            child = Group(Child, share=[go])

        parent = Parent()
        pressed: list[bool] = []
        parent.bind(Parent.go, pressed.append)
        Child.go.fire(parent.child)
        self.assertEqual(pressed, [True])


if __name__ == "__main__":
    unittest.main()
