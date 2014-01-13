# platform.mk
#
# Makefile for cross platform building. It determines the platform and platform
# specific variables like compiler versions. Build rules are also provided.

PLATFORM    := unknown
ifeq ("$(OS)","Windows_NT")
PLATFORM    := windows
WINDOWS_NT  := 1
else
UNAME_S	    := $(shell uname -s)
ifeq ("$(UNAME_S)","Linux")
PLATFORM    := linux
LINUX	    := 1
else
ifeq ("$(UNAME_S)","Darwin")
PLATFORM    := mbsd
MBSD	    := 1
endif
endif
endif

ifndef PROCESSOR
PROCESSOR   := i686
ifeq ("$(OS)","Windows_NT")
ifeq ("$(PROCESSOR_ARCHITECTURE)","AMD64")
PROCESSOR   := x86_64
endif
ifeq ("$(PROCESSOR_ARCHITEW6432)","AMD64")
PROCESSOR   := x86_64
endif
else
UNAME_S	    := $(shell uname -s)
ifeq ("$(UNAME_S)","Linux")
PROCESSOR   := $(shell uname -m)
else
ifeq ("$(UNAME_S)","Darwin")
PROCESSOR   := x86_64
endif
endif
endif
endif

ifeq ("$(PROCESSOR)","x86_64")
AMD64	    := 1
endif

DepFromSrc = $(join $(dir $(1)),$(addsuffix .d,$(basename $(addprefix .,$(notdir $(1))))))

# Set Windows-specific variables. It assumes Cygwin (http://cygwin.org) has
# been installed and that you're running make from within a Cygwin shell.
ifdef WINDOWS_NT
    ifndef MSVCDir
        $(error You must set MSVCDir)
    endif

    # We want to use $MSVCDir in windows short-form format.
    CYGWIN_TOOLROOT	:= $(shell cygpath -u "$(MSVCDir)")
    MSVCDir		:= $(shell cygpath -sw "$(MSVCDir)")
    MSVCDir		:= $(subst \,/,$(MSVCDir))
    
    TOOL_BINPATH	:= $(MSVCDir)/bin
    TOOL_CYGWIN_BINPATH	:= $(CYGWIN_TOOLROOT)/bin
    TOOL_MT_BINPATH	:= $(WIN32_SDK_BIN)
    TOOL_CYG_RC_BINPATH	:= $(CYGWIN_TOOLROOT)/bin
    
    ifdef MSVC_SDK_INCLUDE
	TOOL_INCLUDEPATH 	:= $(MSVC_SDK_INCLUDE)
	TOOL_SDKINCLUDEPATH	:= $(WIN32_SDK_INCLUDE)
    else
	TOOL_INCLUDEPATH 	:= $(MSVCDir)/include
	TOOL_SDKINCLUDEPATH	:= $(MSVCDir)/PlatformSDK/Include
    endif
    
    ifdef MSVC_SDK_LIB
	TOOL_LIBPATH	:= $(MSVC_SDK_LIB)
	TOOL_SDKLIBPATH	:= $(WIN32_SDK_LIB)
    else
	TOOL_LIBPATH	:= $(MSVCDir)/lib
	TOOL_SDKLIBPATH	:= $(MSVCDir)/PlatformSDK/Lib
	ifdef AMD64
	    TOOL_LIBPATH     := $(TOOL_LIBPATH)/amd64
	    TOOL_SDKLIBPATH  := $(TOOL_SDKLIBPATH)/amd64
	endif
    endif
    
    ifdef AMD64
    	TOOL_BINPATH	    := $(TOOL_BINPATH)/amd64
    	TOOL_CYGWIN_BINPATH := $(TOOL_CYGWIN_BINPATH)/amd64
    endif

    # These are specific path environment variables
    # which cl.exe and link.exe use to locate include and library
    # directories respectively.
    export INCLUDE := $(TOOL_INCLUDEPATH);$(TOOL_SDKINCLUDEPATH)
    export LIB := $(TOOL_LIBPATH);$(TOOL_SDKLIBPATH)

    # Make sure that the MSVCDir binaries are available. We put the RC
    # bin path last because it's usually the 32-bit directory.
    export PATH := $(TOOL_CYGWIN_BINPATH):$(PATH):$(TOOL_CYG_RC_BINPATH)

    WARNINGS            += -wd4355 -wd4244 -wd4305 -wd4180
    DEFINES		+= -DNOMINMAX \
			   -D_USE_MATH_DEFINES \
			   -D_HAS_ITERATOR_DEBUGGING=0 \
			   -DWIN32_LEAN_AND_MEAN \
			   -D_WIN32_WINNT=0x0501 \
			    $(NULL)
    DEBUG_FLAGS		+= -Od -Z7
ifeq (full,$(strip $(debug)))
    DEBUG_FLAGS		+= -MDd -RTC1
else
    DEBUG_FLAGS		+= -MD
endif
    OPTIMIZE_FLAGS	+= -MD -O2 -DNDEBUG
    CXX			:= $(TOOL_BINPATH)/cl -nologo
    CXXFLAGS		+= -bigobj -EHsc -Zc:forScope -TP \
                           $(WARNINGS) \
			   $(DEFINES) \
			   -I "$(TOOL_INCLUDEPATH)" \
			   -I "$(TOOL_SDKINCLUDEPATH)" \
			   $(NULL)
    CXXOUTPUT		:= -Fo
    LINK		:= $(TOOL_BINPATH)/link -nologo
    LDFLAGS		+= -VERSION:$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION) \
                           -libpath:"$(TOOL_LIBPATH)" \
			   -libpath:"$(TOOL_SDKLIBPATH)" \
			   $(NULL)
    LDOUTPUT		:= -out:
    LDDIROPT		:= -libpath:
    AR			:= $(TOOL_BINPATH)/lib -nologo
    AROUTPUT		:= -out:
    ARFLAGS		:= 
    LN			:= /usr/local/bin/winln -sv

    EXE_SUFFIX		:= .exe
    STATICLIB_PREFIX	:= lib
    STATICLIB_SUFFIX	:= .lib
    SHAREDLIB_PREFIX	:=
    SHAREDLIB_SUFFIX	:= .dll
    IMPORTLIB_PREFIX	:=
    IMPORTLIB_SUFFIX	:= .lib
    VERSIONED_LIBS	:= no

    MAKEDEP_SYSINCLUDES := -S "$(TOOL_INCLUDEPATH)" -S "$(TOOL_SDKINCLUDEPATH)"
    ifneq (,$(strip $(HFS)))
	MAKEDEP_SYSINCLUDES += -S "$(HFS)/custom/include"
	MAKEDEP_SYSINCLUDES += -S "$(HFS)/toolkit/include"
    endif
    SELF_DIR		:= $(dir $(lastword $(MAKEFILE_LIST)))
    MAKEDEPEND		:= $(SELF_DIR)/clmakedep.py $(MAKEDEP_SYSINCLUDES)

    define BuildSharedLibrary
	@echo "Building $@ because of $(call list_deps)"; \
	 $(LINK) -DLL $(LDFLAGS) $(LDOUTPUT)$@ $^ $(1)
    endef

    define BuildExecutable
	@echo "Building $@ because of $(call list_deps)"; \
	 $(LINK) $(LDFLAGS) $(LDOUTPUT)$@ $(1)
    endef

    define CompileCXX
	@echo "Building $@ because of $(call list_deps)"; \
	 $(CXX) -c $(CXXFLAGS) $(1) $(CXXOUTPUT)$@ $< \
	 && echo "... making deps" \
	 && $(MAKEDEPEND) -- $(CXXFLAGS) $(1) -- $< > $(call DepFromSrc,$<)
    endef

else

    # Generic flags for non-Windows

    DEFINES		+=
    DEBUG_FLAGS		+= -g
    OPTIMIZE_FLAGS	+= -O3 -DNDEBUG
    CXX_WARNFLAGS	:= -fmessage-length=0 \
			   -Wall -W \
			   -Wno-sign-compare \
			   -Wno-parentheses \
			   -Wno-unused-parameter \
			   -Wno-reorder \
			   -Wnon-virtual-dtor \
			   -Woverloaded-virtual \
			   $(NULL)
    CXXFLAGS		+= $(DEFINES) -std=c++0x -pthread -fPIC \
			   -fvisibility=hidden -fvisibility-inlines-hidden \
			   $(CXX_WARNFLAGS) \
			   $(NULL)
    CXXOUTPUT		:= -o
ifdef MBSD
    LINK		= $(CXX) $(CXXFLAGS) -dynamiclib
else
    LINK		= $(CXX) $(CXXFLAGS) -shared
    LDFLAGS		= -Wl,-rpath,'$$ORIGIN/.'
endif
    LDLIBS		+= -ldl -lm
ifdef LINUX
    LDLIBS		+= -lrt
endif

    LDOUTPUT		:= -o
    LDDIROPT		:= -L
    AROUTPUT		:=
    MAKEDEPEND		= $(CXX) -O0 -MM $(1) -MT `echo $(1) | sed 's%\.[^.]*%.o%'`
    LN			:= /bin/ln -sf

    EXE_SUFFIX		:=
    STATICLIB_PREFIX	:= lib
    STATICLIB_SUFFIX	:= .a
    SHAREDLIB_PREFIX	:= lib
ifdef MBSD
    SHAREDLIB_SUFFIX	:= .dylib
else
    SHAREDLIB_SUFFIX	:= .so
endif
    IMPORTLIB_PREFIX	:= $(SHAREDLIB_PREFIX)
    IMPORTLIB_SUFFIX	:= $(SHAREDLIB_SUFFIX)
    VERSIONED_LIBS	:=

    define BuildSharedLibrary
	@echo "Building $@ because of $(call list_deps)"; \
	 $(LINK) $(LDFLAGS) $(LDOUTPUT) $@ $^ $(1) $(LDLIBS)
    endef

    define BuildExecutable
	@echo "Building $@ because of $(call list_deps)"; \
	 $(CXX) $(CXXFLAGS) $(CXXOUTPUT) $@ $(1) $(LDLIBS)
    endef

    define CompileCXX
	@echo "Building $@ because of $(call list_deps)"; \
	 $(CXX) -c $(CXXFLAGS) -MD -MP -MF .$(notdir $(@:.o=.d)) $(1) \
                $(CXXOUTPUT) $@ $<
    endef

endif

# Set OSX-specific variables.
ifdef MBSD

    DARWIN_OS_MAJOR_VER := $(basename $(basename $(shell uname -r)))
    ifeq ($(DARWIN_OS_MAJOR_VER),10)
        # Snow Leopard
        MACOSX_SDK := MacOSX10.6
    else ifeq ($(DARWIN_OS_MAJOR_VER),11)
        # Lion
        MACOSX_SDK := MacOSX10.7
    else ifeq ($(DARWIN_OS_MAJOR_VER),12)
        # Mountain Lion
        MACOSX_SDK := MacOSX10.8
    else ifeq ($(DARWIN_OS_MAJOR_VER),13)
	# Mavericks
	MACOSX_SDK := MacOSX10.9
    else
        $(error Unknown MacOSX Darwin major version $(DARWIN_OS_MAJOR_VER))
    endif

    MACOSX_SDK_PATH := /Developer/SDKs/$(MACOSX_SDK).sdk
    ifeq ($(wildcard $(MACOSX_SDK_PATH)),)
        # The SDK moved to /Applications/Xcode.app as of XCode 4.3.1.
        MACOSX_SDK_PATH := /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/$(MACOSX_SDK).sdk
    endif

    # If MBSD_TARGET_VERSION is set, then we use it as the mininum target SDK
    # version.
    ifdef MBSD_TARGET_VERSION
        MACOSX_TARGET_VERSION := -mmacosx-version-min=$(MBSD_TARGET_VERSION)
    else
        MACOSX_TARGET_VERSION :=
    endif

    DARWIN_FLAGS    := -arch $(PROCESSOR) \
                       -isysroot $(MACOSX_SDK_PATH) \
                       $(MACOSX_TARGET_VERSION)

    CC              := $(CC) $(DARWIN_FLAGS)
    CXX             := $(CXX) $(DARWIN_FLAGS)

endif


ifdef PLATFORM_ENABLED
# Rule for testing platform detection
platform:
	@echo PLATFORM=$(PLATFORM)
	@echo LINUX=$(LINUX)
	@echo WINDOWS_NT=$(WINDOWS_NT)
	@echo MSVCDir=$(MSVCDir)
	@echo MBSD=$(MBSD)
	@echo AMD64=$(AMD64)
	@echo CXX=$(CXX)
	@echo CXXFLAGS=$(CXXFLAGS)
endif
