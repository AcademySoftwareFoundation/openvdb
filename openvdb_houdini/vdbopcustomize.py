import hou
import sys

sopCategory = hou.sopNodeTypeCategory()

dsoPaths = sys.argv[1:]

# load all OpenVDB SOP DSOs
for dsoPath in dsoPaths:
    sopCategory.loadDSO(dsoPath)

# build dictionaries of native and aswf SOPs with a hidden state
nativeSOPs = dict()
aswfSOPs = dict()

for (nodeName, nodeType) in sopCategory.nodeTypes().iteritems():
    if 'vdb' not in nodeName.lower():
        continue
    if nodeType.source() != hou.nodeTypeSource.CompiledCode:
        continue
    isASWF = nodeType.sourcePath() in dsoPaths
    isHidden = nodeType.hidden()

    if isASWF:
        aswfSOPs[nodeName] = isHidden
    else:
        nativeSOPs[nodeName] = isHidden

# ensure SOPs are listed alphabetically
aswfKeys = aswfSOPs.keys()
nativeKeys = nativeSOPs.keys()

aswfKeys.sort()
nativeKeys.sort()

# print ophide statements for all SOPs, disable with comments
# those that need not be hidden

opHide = '{comment}ophide Sop {name}'

print '// Native VDB SOPs'

for name in nativeKeys:
    hidden = nativeSOPs[name]
    print opHide.format(name=name, comment=('' if hidden else '// '))

print ''
print '// ASWF VDB SOPs'

for name in aswfKeys:
    hidden = aswfSOPs[name]
    print opHide.format(name=name, comment=('' if hidden else '// '))
