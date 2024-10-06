# config.py

def can_build(env, platform):
    if platform in ("linuxbsd", "windows", "android", "macos"):
        return True
    else:
        # not supported on these platforms
        return False

def configure(env):
    pass
