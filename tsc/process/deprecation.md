**Deprecation Strategy for OpenVDB**

OpenVDB is committed to supporting the current and previous two years of the
[VFX Reference Platform](http://www.vfxplatform.com/) and all releases of
Houdini and Maya based on those versions of the platform.

For example, as of writing it is March 2019 which means VFX Reference Platform
2019, 2018 and 2017

This infers the following support:

* OpenVDB ABI=4, ABI=5 and ABI=6
* C++11 and C++14
* Houdini 16.5, 17.0 and 17.5

In January 2020, OpenVDB will drop support for VFX Reference Platform 2017.
This will mean eliminating support for Houdini 16.5, ABI=4 and C++11.
