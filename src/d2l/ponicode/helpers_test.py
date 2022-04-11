import helpers


class Test_Helpers_Set_axes:
    def test_set_axes_1(self):
        result = helpers.set_axes(
            0, "label_2", "label_1", "Nile Crocodile", "Michael", 0.1, 0.5, "ISO 9001"
        )

    def test_set_axes_2(self):
        result = helpers.set_axes(
            1, "ISO 9001", "label_2", "Nile Crocodile", "Edmond", 0.1, 0.5, "ISO 22000"
        )

    def test_set_axes_3(self):
        result = helpers.set_axes(
            1,
            "ISO 9001",
            "AOP",
            "Australian Freshwater Crocodile",
            "Jean-Philippe",
            0.5,
            1.0,
            "ISO 9001",
        )

    def test_set_axes_4(self):
        result = helpers.set_axes(
            3,
            "ISO 9001",
            "ISO 9001",
            "Dwarf Crocodile",
            "Jean-Philippe",
            2.0,
            0.1,
            "label_3",
        )

    def test_set_axes_5(self):
        result = helpers.set_axes(
            0,
            "label_2",
            "ISO 9001",
            "Australian Freshwater Crocodile",
            "Jean-Philippe",
            0.1,
            1.0,
            "label_3",
        )

    def test_set_axes_6(self):
        result = helpers.set_axes(0, "", "", "", "", 0, 0, "")
