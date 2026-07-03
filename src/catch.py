from typing import Callable, TypeVar, Any
from json import JSONDecodeError

T = TypeVar("T")


def catch(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        try:
            return fn(*args, **kwargs)
        except JSONDecodeError as e:
            raise JSONDecodeError(f"JSON error: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: File missing: {e}")
        except ImportError as e:
            raise ImportError(f"Import error: {e}")
        except KeyboardInterrupt:
            print("\nKeyboard interrupt. Bye!")
            quit(0)
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
    except Exception as e:
        print(e)
        quit(1)
