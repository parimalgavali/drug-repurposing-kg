from api.ratelimit import RateLimiter


def test_window_slides_per_key():
    now = [0.0]
    r = RateLimiter(2, 60, clock=lambda: now[0])
    assert r.hit("a") == 0 and r.hit("a") == 0
    assert r.hit("a") == 60 and r.hit("b") == 0
    now[0] = 60.0
    assert r.hit("a") == 0


def test_zero_limit_disables():
    r = RateLimiter(0, 60)
    assert all(r.hit("a") == 0 for _ in range(100))
