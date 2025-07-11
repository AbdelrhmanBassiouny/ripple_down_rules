import ripple_down_rules
from . import overrides

# Apply all registered overrides
for name, func in overrides._overrides:
    setattr(ripple_down_rules, name, func)
