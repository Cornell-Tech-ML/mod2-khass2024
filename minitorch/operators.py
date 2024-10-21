"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return x if x > 0 else 0


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg."""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element in an iterable.

    Args:
    ----
        fn: A function that takes a float and returns a float.
        ls: An iterable of floats.

    Returns:
    -------
        An iterable of floats resulting from applying fn to each element in ls.

    """
    return (fn(x) for x in ls)


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Apply a function to pairs of elements from two iterables.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        ls1: The first iterable of floats.
        ls2: The second iterable of floats.

    Returns:
    -------
        An iterable of floats resulting from applying fn to pairs of elements from ls1 and ls2.

    """
    return (fn(x, y) for x, y in zip(ls1, ls2))


def reduce(fn: Callable[[float, float], float], ls: Iterable[float]) -> float:
    """Reduce an iterable to a single value using a binary function.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        ls: An iterable of floats.

    Returns:
    -------
        A float resulting from repeatedly applying fn to the elements of ls.

    Raises:
    ------
        ValueError: If ls is empty.

    """
    iterator = iter(ls)
    try:
        result = next(iterator)
    except StopIteration:
        raise ValueError("Cannot reduce an empty iterable")
    for x in iterator:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in an iterable.

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        An iterable of floats where each element is the negation of the corresponding element in ls.

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two iterables.

    Args:
    ----
        ls1: The first iterable of floats.
        ls2: The second iterable of floats.

    Returns:
    -------
        An iterable of floats where each element is the sum of the corresponding elements in ls1 and ls2.

    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Calculate the sum of all elements in an iterable.

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        The sum of all elements in ls, or 0 if ls is empty.

    """
    try:
        return reduce(add, ls)
    except ValueError:
        return 0.0


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in an iterable.

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        The product of all elements in ls.

    """
    return reduce(mul, ls)
