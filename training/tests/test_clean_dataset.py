from carclass.clean_dataset import infer_view_and_class


def test_infer_view_and_class():
    assert infer_view_and_class("car_brands/toyota_avanza/front/a.jpg") == ("toyota_avanza", "front")
    assert infer_view_and_class("car_brands/bmw/single.jpg") == ("bmw", "unknown")
