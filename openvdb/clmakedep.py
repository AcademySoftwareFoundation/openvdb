#!/usr/bin/env python
"""
    clmakedep.py

    Uses Visual C++ to generate Makefile dependencies.

    Example Makefile rule:
	depend:
		./clmakedep.py -- $(CFLAGS) -- $(SRCS)
"""

import ntpath as ospath
import os
import re
import subprocess
import sys

SYSTEM_INCLUDES = []

def makePath(words, base=None):
    if base:
        path = base + '/'
    else:
        path = ''
    path += ' '.join(words).replace('\\','/')
    if path.find(' ') >= 0:
        path = subprocess.Popen(
                    ['cygpath', '-ams', path],
                    stdout=subprocess.PIPE).communicate()[0].strip()
    return path

def writeRule(out, target, deps):
    # remove duplicates in deps, except for the first one (ie. the src)
    first_dep = deps[0]
    deps = list(set(deps[1:]))
    deps.insert(0, first_dep)
    # remove items found in system includes, do case-insensitive match
    deps = [d for d in deps \
            if not len([s for s in SYSTEM_INCLUDES \
                        if d.lower().startswith(s.lower() + '/')])]
    # write rule
    out.write(target + ': ')
    for path in deps:
	out.write('\t' + path.replace(' ', '\\\\ ') + ' \\\n')
    out.write('#\n')
    # return actual deps used
    return deps

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option(
            "-o", "", dest="objsuffix",
            action="store", default=".o",
            metavar="SUFFIX",
            help="Specify object file suffix [default: %default]")
    parser.add_option(
            "-S", "", dest="sysincludes",
            action="append",
            metavar="INCL_DIR",
            help="Specify system include directory to omit from output")
    options, args_list = parser.parse_args()

    objsuffix = options.objsuffix
    global SYSTEM_INCLUDES
    SYSTEM_INCLUDES = [makePath([x]) for x in options.sysincludes]

    out = sys.stdout

    found_double_hyphen = False
    dirs = []
    includes = []
    cmd = ['cl', '-nologo', '-TP', '-showIncludes', '-Zs', '-w']
    for i,arg in enumerate(args_list):
	# simply note whether we've see '--' yet
	if arg == '--':
	    found_double_hyphen = True
	    continue
	# always add the arg to cmd to be passed to cl
	cmd.append(arg)
	if found_double_hyphen:
            dirs.append(ospath.dirname(arg))
	    continue
	# only do -I parsing before we've found '--'
	if arg == '-I':
	    includes.append(makePath([args_list[i+1]]))
    #print "cmd:", cmd
    #print "includes:", includes
    #print "dirs:", dirs

    i = 0
    target = None
    deps = []
    files = []
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    for line in output.splitlines():
        words = line.split()
	if len(words) < 1:
	    continue
	if words[0] != 'Note:':
	    # flush rule
	    if target and len(deps) > 0:
	        files.extend(writeRule(out, target, deps))
		deps = []
	    # create target and make the source file as the first dep
	    src = makePath(words, base=dirs[i])
	    target = ospath.splitext(ospath.basename(src))[0] + objsuffix
	    deps = [src]
	    continue
	# record dependency
	deps.append(makePath(words[3:]))
    # flush rule
    if target and len(deps) > 0:
	files.extend(writeRule(out, target, deps))
    # output list of deps as targets to handle when they get deleted
    files = list(set(files)) # remove duplicates
    for path in files:
	out.write(path.replace(' ', '\\\\ ') + ':\n')


if __name__ == '__main__':
    main()

