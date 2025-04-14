class ScopedDict:
    """Dictionary that supports "scopes"

    Different scopes may contain variables with different names.
    You can use this to prevent variables in different scopes
    overriding each other. For example:

    bindings["x"] = 1
    ...
    bindings.push_scope()
    bindings["x"] = 2
    print(bindings["x"})  # prints 2
    bindings.pop_scope()

    print(bindings["x"})  # prints 1

    Variables in outer scopes remain visible in inner ones.
    For example:

    bindings["x"] = 1
    bindings.push_scope()
    print(bindings["x"])

    This class might be useful to prevent function parameters being
    visible in outer scopes. For example:

    def foo(x: Mat(1,1)) {
        print(x);
    }
    foo(5);
    print(x); # Error!
    """

    dicts: [dict]

    def __init__(self):
        """Initialize the ScopedDict with a single empty scope."""
        self.dicts = [{}]

    def __getitem__(self, key):
        """Retrieve the value associated with the key from the nearest scope.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key.

        Raises:
            AssertionError: If the key is not found in any scope.
        """
        for d in reversed(self.dicts):
            if d.get(key, None) is not None:
                return d[key]
        assert False, f"dictionary did not contain key {key}"

    def __setitem__(self, key, value):
        """Set the value for a key in the current (innermost) scope.

        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        self.dicts[-1][key] = value

    def get(self, key):
        """Retrieve the value associated with the key from the nearest scope.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key, or None if the key is not found.
        """
        for d in reversed(self.dicts):
            if d.get(key, None) is not None:
                return d[key]
        return None

    def push_scope(self):
        """Create a new (inner) scope."""
        self.dicts.append({})

    def pop_scope(self):
        """Remove the current (innermost) scope.

        Raises:
            IndexError: If there are no scopes to pop.
        """
        self.dicts.pop()
