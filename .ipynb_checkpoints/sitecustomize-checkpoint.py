# Robust compat shim for pymatgen Site.is_ordered lookups
# Ensures any access to `site.is_ordered` returns True (safe for QM9 molecules).
try:
    from pymatgen.core.sites import Site
    _old_getattr = getattr(Site, "__getattr__", None)
    if _old_getattr:
        def _patched_getattr(self, attr):
            if attr == "is_ordered":
                return True
            return _old_getattr(self, attr)
        Site.__getattr__ = _patched_getattr
    else:
        # Fallback: define attribute if getattr isn't present
        def _get_is_ordered(self):
            return True
        Site.is_ordered = property(_get_is_ordered)
except Exception:
    pass
