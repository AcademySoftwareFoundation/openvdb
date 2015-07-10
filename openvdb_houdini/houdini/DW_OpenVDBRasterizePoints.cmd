# Automatically generated script: Monday June 29, 10:50 2015

\set noalias = 1
#
#  Creation script for DW_OpenVDBRasterizePoints operator
#

if ( "$arg1" == "" ) then
    echo This script is intended as a creation script
    exit
endif

# Node $arg1 (Sop/DW_OpenVDBRasterizePoints)
opspareds '    parm { 	name	"pointgroup" 	baseparm 	export	none     }     parm { 	name	"voxelsize" 	baseparm 	export	none     }     parm { 	name	"createdensity" 	baseparm 	export	none     }     parm { 	name	"compositing" 	baseparm 	export	none     }     parm { 	name	"densityscale" 	baseparm 	export	none     }     parm { 	name	"particlescale" 	baseparm 	export	none     }     parm { 	name	"solidratio" 	baseparm 	export	none     }     parm { 	name	"attributes" 	baseparm 	export	none     }     parm { 	name	"noiseheading" 	baseparm 	export	none     }     parm { 	name	"modeling" 	baseparm 	export	none     }     parm { 	name	"lookup" 	label	"Noise Lookup" 	type	integer 	default	{ "0" } 	menu	{ 	    "0"	"Displacement" 	    "1"	"World Space" 	    "2"	"Local Space" 	    "3"	"Unit Space" 	} 	range	{ 0 10 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"treatment" 	label	"Noise Treatment" 	type	integer 	default	{ "0" } 	menu	{ 	    "0"	"Abs" 	    "1"	"1 - Abs" 	    "2"	"Clamp" 	} 	range	{ 0 10 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"amp" 	label	"Amplitude" 	type	float 	default	{ "1" } 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"excavation" 	label	"Excavation" 	type	float 	default	{ "0.07" } 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"falloff" 	label	"Falloff" 	type	float 	default	{ "0.1" } 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"freq" 	label	"Frequency" 	type	float 	default	{ "1.74" } 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"rough" 	label	"Roughness" 	type	float 	default	{ "0.5" } 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"ocataves" 	label	"Ocataves" 	type	integer 	default	{ "2" } 	range	{ 0 10 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"offset" 	label	"Offset" 	type	float 	size	3 	default	{ "0" "0" "0" } 	range	{ 0 10 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"billowing" 	label	"Billowing Motion" 	type	toggle 	default	{ "off" } 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"billowingspeed" 	label	"Billowing Speed" 	type	float 	default	{ "0.1" } 	disablewhen	"{ billowing != 1 }" 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"swirling" 	label	"Swirling Motion" 	type	toggle 	default	{ "off" } 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     }     parm { 	name	"swirlingspeed" 	label	"Swirling Speed" 	type	float 	default	{ "0.1" } 	disablewhen	"{ billowing != 1 }" 	range	{ 0 1 } 	export	none 	parmtag	{ "parmvop" "1" } 	parmtag	{ "shaderparmcontexts" "cvex" }     } ' $arg1
opparm -V 14.0.254 $arg1 pointgroup ( "" ) voxelsize ( 0.10000000000000001 ) createdensity ( on ) compositing ( max ) densityscale ( 1 ) particlescale ( 1 ) solidratio ( 0 ) attributes ( "" ) noiseheading ( ) modeling ( off ) lookup ( 0 ) treatment ( 0 ) amp ( 1 ) excavation ( 0.070000000000000007 ) falloff ( 0.10000000000000001 ) freq ( 1.74 ) rough ( 0.5 ) ocataves ( 2 ) offset ( 0 0 0 ) billowing ( off ) billowingspeed ( 0.10000000000000001 ) swirling ( off ) swirlingspeed ( 0.10000000000000001 )
chlock $arg1 -*
chautoscope $arg1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 $arg1
opexprlanguage -s hscript $arg1
opuserdata -n '___Version___' -v '14.0.254' $arg1

opcf $arg1

# Network Box apply_billowing_motion
nbadd apply_billowing_motion
nblocate -x -34.9668 -y 4.6532 apply_billowing_motion
nbsize -w 9.83121 -h 5.57119 apply_billowing_motion
nbset  -m off apply_billowing_motion
nbcolor -c 0.8 1 0.8 apply_billowing_motion

# Node density_scale (Vop/bind)
opadd -e -n bind density_scale
oplocate -x 19.233899999999998 -y 5.9932499999999997 density_scale
opspareds "" density_scale
opparm -V 14.0.254 density_scale parmname ( output ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( on ) exportcontext ( cvex )
chlock density_scale -*
chautoscope density_scale -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 density_scale
opset -d on -r on -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on density_scale
opwire -n clamp1 -0 density_scale
opexprlanguage -s hscript density_scale
opuserdata -n '___Version___' -v '14.0.254' density_scale

# Node Inputs (Vop/subnet)
opadd -e -n subnet Inputs
oplocate -x -38.680100000000003 -y 10.726699999999999 Inputs
opspareds "" Inputs
opparm -V 14.0.254 Inputs
chlock Inputs -*
chautoscope Inputs -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 Inputs
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Inputs
opexprlanguage -s hscript Inputs
opuserdata -n '___Version___' -v '14.0.254' Inputs
opcf Inputs

# Network Box customize_particle_orientation
nbadd customize_particle_orientation
nblocate -x -23.4388 -y 5.8788 customize_particle_orientation
nbsize -w 18.4232 -h 10.27 customize_particle_orientation
nbset  -m off customize_particle_orientation
nbcolor -c 0.52 0.52 0.52 customize_particle_orientation

# Network Box spherical_coords
nbadd spherical_coords
nblocate -x -0.816842 -y 4.88635 spherical_coords
nbsize -w 7.14363 -h 3.35845 spherical_coords
nbset  -m off spherical_coords
nbcolor -c 0.52 0.52 0.52 spherical_coords

# Node voxel_pos (Vop/bind)
opadd -e -n bind voxel_pos
oplocate -x -23.038799999999998 -y 6.4199999999999999 voxel_pos
opspareds "" voxel_pos
opparm -V 14.0.254 voxel_pos parmname ( voxelpos ) parmtype ( float3 ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock voxel_pos -*
chautoscope voxel_pos -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 voxel_pos
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on voxel_pos
nbop customize_particle_orientation add voxel_pos
opexprlanguage -s hscript voxel_pos
opuserdata -n '___Version___' -v '14.0.254' voxel_pos

# Node voxel_size (Vop/bind)
opadd -e -n bind voxel_size
oplocate -x -0.23407800000000001 -y 2.9261499999999998 voxel_size
opspareds "" voxel_size
opparm -V 14.0.254 voxel_size parmname ( voxelsize ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock voxel_size -*
chautoscope voxel_size -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 voxel_size
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on voxel_size
opexprlanguage -s hscript voxel_size
opuserdata -n '___Version___' -v '14.0.254' voxel_size

# Node particle_center (Vop/bind)
opadd -e -n bind particle_center
oplocate -x -22.822299999999998 -y 4.5960200000000002 particle_center
opspareds "" particle_center
opparm -V 14.0.254 particle_center parmname ( pcenter ) parmtype ( vector ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_center -*
chautoscope particle_center -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_center
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_center
opexprlanguage -s hscript particle_center
opuserdata -n '___Version___' -v '14.0.254' particle_center

# Node particle_radius (Vop/bind)
opadd -e -n bind particle_radius
oplocate -x 1.9589000000000001 -y 1.75379 particle_radius
opspareds "" particle_radius
opparm -V 14.0.254 particle_radius parmname ( pradius ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_radius -*
chautoscope particle_radius -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_radius
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_radius
opexprlanguage -s hscript particle_radius
opuserdata -n '___Version___' -v '14.0.254' particle_radius

# Node particle_index (Vop/bind)
opadd -e -n bind particle_index
oplocate -x 3.0087299999999999 -y 0.87815100000000001 particle_index
opspareds "" particle_index
opparm -V 14.0.254 particle_index parmname ( pindex ) parmtype ( int ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_index -*
chautoscope particle_index -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_index
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_index
opexprlanguage -s hscript particle_index
opuserdata -n '___Version___' -v '14.0.254' particle_index

# Node Frame (Vop/bind)
opadd -e -n bind Frame
oplocate -x 7.6593600000000004 -y 0.0014801 Frame
opspareds "" Frame
opparm -V 14.0.254 Frame parmname ( Frame ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock Frame -*
chautoscope Frame -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 Frame
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Frame
opexprlanguage -s hscript Frame
opuserdata -n '___Version___' -v '14.0.254' Frame

# Node Time (Vop/bind)
opadd -e -n bind Time
oplocate -x 9.4067399999999992 -y -0.90951899999999997 Time
opspareds "" Time
opparm -V 14.0.254 Time parmname ( Time ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock Time -*
chautoscope Time -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 Time
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Time
opexprlanguage -s hscript Time
opuserdata -n '___Version___' -v '14.0.254' Time

# Node TimeInc (Vop/bind)
opadd -e -n bind TimeInc
oplocate -x 10.5724 -y -1.93123 TimeInc
opspareds "" TimeInc
opparm -V 14.0.254 TimeInc parmname ( TimeInc ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock TimeInc -*
chautoscope TimeInc -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 TimeInc
opset -d on -r on -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on TimeInc
opexprlanguage -s hscript TimeInc
opuserdata -n '___Version___' -v '14.0.254' TimeInc

# Node suboutput1 (Vop/suboutput)
opadd -e -n suboutput suboutput1
oplocate -x 21.3111 -y 2.9153799999999999 suboutput1
opspareds "" suboutput1
opparm -V 14.0.254 suboutput1 name1 ( P ) label1 ( 'World space pos' ) name2 ( CP ) label2 ( 'Origin centered pos (P - pcenter)' ) name3 ( SP ) label3 ( 'Spherical coordinate' ) name4 ( voxelsize ) label4 ( 'Voxel size' ) name5 ( "" ) label5 ( 'Particle center' ) name6 ( "" ) label6 ( 'Particle radius' ) name7 ( "" ) label7 ( 'Particle index' ) name8 ( "" ) label8 ( "" ) name9 ( "" ) label9 ( "" ) name10 ( "" ) label10 ( "" ) name11 ( "" ) label11 ( "" ) name12 ( "" ) label12 ( "" ) name13 ( "" ) label13 ( "" ) name14 ( "" ) label14 ( "" ) name15 ( "" ) label15 ( "" ) name16 ( "" ) label16 ( "" ) name17 ( "" ) label17 ( "" ) name18 ( "" ) label18 ( "" ) name19 ( "" ) label19 ( "" ) name20 ( "" ) label20 ( "" ) name21 ( "" ) label21 ( "" ) name22 ( "" ) label22 ( "" ) name23 ( "" ) label23 ( "" ) name24 ( "" ) label24 ( "" ) name25 ( "" ) label25 ( "" ) name26 ( "" ) label26 ( "" ) name27 ( "" ) label27 ( "" ) name28 ( "" ) label28 ( "" ) name29 ( "" ) label29 ( "" ) name30 ( "" ) label30 ( "" ) name31 ( "" ) label31 ( "" ) name32 ( "" ) label32 ( "" ) name33 ( "" ) label33 ( "" ) name34 ( "" ) label34 ( "" ) name35 ( "" ) label35 ( "" ) name36 ( "" ) label36 ( "" ) name37 ( "" ) label37 ( "" ) name38 ( "" ) label38 ( "" ) name39 ( "" ) label39 ( "" ) name40 ( "" ) label40 ( "" ) name41 ( "" ) label41 ( "" ) name42 ( "" ) label42 ( "" ) name43 ( "" ) label43 ( "" ) name44 ( "" ) label44 ( "" ) name45 ( "" ) label45 ( "" ) name46 ( "" ) label46 ( "" ) name47 ( "" ) label47 ( "" ) name48 ( "" ) label48 ( "" ) name49 ( "" ) label49 ( "" ) name50 ( "" ) label50 ( "" ) name51 ( "" ) label51 ( "" ) name52 ( "" ) label52 ( "" ) name53 ( "" ) label53 ( "" ) name54 ( "" ) label54 ( "" ) name55 ( "" ) label55 ( "" ) name56 ( "" ) label56 ( "" ) name57 ( "" ) label57 ( "" ) name58 ( "" ) label58 ( "" ) name59 ( "" ) label59 ( "" ) name60 ( "" ) label60 ( "" ) name61 ( "" ) label61 ( "" ) name62 ( "" ) label62 ( "" ) name63 ( "" ) label63 ( "" ) name64 ( "" ) label64 ( "" )
chlock suboutput1 -*
chautoscope suboutput1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 suboutput1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on suboutput1
opwire -n P -0 suboutput1
opwire -n CP -1 suboutput1
opwire -n spherical_coord -2 suboutput1
opwire -n voxel_size -3 suboutput1
opwire -n particle_center -4 suboutput1
opwire -n particle_radius -5 suboutput1
opwire -n particle_index -6 suboutput1
opwire -n Frame -7 suboutput1
opwire -n Time -8 suboutput1
opwire -n TimeInc -9 suboutput1
opexprlanguage -s hscript suboutput1
opuserdata -n '___Version___' -v '14.0.254' suboutput1

# Node subinput1 (Vop/subinput)
opadd -e -n subinput subinput1
oplocate -x -19.8491 -y 1.6854800000000001 subinput1
opspareds "" subinput1
opparm -V 14.0.254 subinput1
chlock subinput1 -*
chautoscope subinput1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 subinput1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on subinput1
opexprlanguage -s hscript subinput1
opuserdata -n '___Version___' -v '14.0.254' subinput1

# Node length1 (Vop/length)
opadd -e -n length length1
oplocate -x -0.075307700000000005 -y 7.2918000000000003 length1
opspareds "" length1
opparm length1 signature ( default ) vec ( 1 1 1 ) vec_p ( 1 1 1 ) vec_n ( 1 1 1 ) vec_v4 ( 1 1 1 1 ) vec_uv ( 1 1 1 ) vec_up ( 1 1 1 ) vec_un ( 1 1 1 )
chlock length1 -*
chautoscope length1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 length1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on length1
opwire -n CP -0 length1
nbop spherical_coords add length1
opexprlanguage -s hscript length1
opuserdata -n '___Version___' -v '' length1

# Node split (Vop/vectofloat)
opadd -e -n vectofloat split
oplocate -x -0.41684199999999999 -y 5.7895899999999996 split
opspareds "" split
opparm split signature ( default ) vec ( 0 0 0 ) vec_p ( 0 0 0 ) vec_n ( 0 0 0 ) vec_c ( 0 0 0 ) vec_uv ( 0 0 0 ) vec_up ( 0 0 0 ) vec_un ( 0 0 0 ) vec_uc ( 0 0 0 )
chlock split -*
chautoscope split -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 split
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on split
opwire -n CP -0 split
nbop spherical_coords add split
opexprlanguage -s hscript split
opuserdata -n '___Version___' -v '' split

# Node spherical_coord (Vop/floattovec)
opadd -e -n floattovec spherical_coord
oplocate -x 4.79718 -y 6.3563000000000001 spherical_coord
opspareds "" spherical_coord
opparm spherical_coord signature ( default ) fval1 ( 0 ) fval2 ( 0 ) fval3 ( 0 ) fval1_uv ( 0 ) fval2_uv ( 0 ) fval3_uv ( 0 )
chlock spherical_coord -*
chautoscope spherical_coord -*
opcolor -c 0.40000000596046448 1 1 spherical_coord
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on spherical_coord
opwire -n length1 -0 spherical_coord
opwire -n atan1 -1 spherical_coord
opwire -n trig1 -2 spherical_coord
nbop spherical_coords add spherical_coord
opexprlanguage -s hscript spherical_coord
opuserdata -n '___Version___' -v '' spherical_coord

# Node atan1 (Vop/atan)
opadd -e -n atan atan1
oplocate -x 2.3095300000000001 -y 6.3746099999999997 atan1
opspareds "" atan1
opparm atan1 y ( 0 ) x ( 1 )
chlock atan1 -*
chautoscope atan1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 atan1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on atan1
opwire -n -o 1 split -0 atan1
opwire -n split -1 atan1
nbop spherical_coords add atan1
opexprlanguage -s hscript atan1
opuserdata -n '___Version___' -v '' atan1

# Node divide1 (Vop/divide)
opadd -e -n divide divide1
oplocate -x 1.6752499999999999 -y 5.2863499999999997 divide1
opspareds "" divide1
opparm -V 14.0.254 divide1
chlock divide1 -*
chautoscope divide1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 divide1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on divide1
opwire -n -o 2 split -0 divide1
opwire -n length1 -1 divide1
nbop spherical_coords add divide1
opexprlanguage -s hscript divide1
opuserdata -n '___Version___' -v '14.0.254' divide1

# Node trig1 (Vop/trig)
opadd -e -n trig trig1
oplocate -x 3.6118700000000001 -y 5.2863600000000002 trig1
opspareds "" trig1
opparm trig1 signature ( default ) func ( vop_acos ) rad ( 0 ) rad_v ( 0 0 0 ) rad_p ( 0 0 0 ) rad_n ( 0 0 0 ) rad_c ( 0 0 0 ) rad_v4 ( 0 0 0 0 ) rad_uf ( 0 ) rad_uv ( 0 0 0 ) rad_up ( 0 0 0 ) rad_un ( 0 0 0 ) rad_uc ( 0 0 0 ) freq ( 1 ) offset ( 0 )
chlock trig1 -*
chautoscope trig1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 trig1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on trig1
opwire -n divide1 -0 trig1
nbop spherical_coords add trig1
opexprlanguage -s hscript trig1
opuserdata -n '___Version___' -v '' trig1

# Node get_ang_attribute (Vop/getattrib)
opadd -e -n getattrib get_ang_attribute
oplocate -x -21.6783 -y 10.285 get_ang_attribute
opspareds "" get_ang_attribute
opparm get_ang_attribute signature ( f ) opinput ( opinput:0 ) file ( '$HH/geo/defgeo.bgeo' ) atype ( point ) attrib ( ang ) i1 ( 0 ) i2 ( 0 )
chlock get_ang_attribute -*
chautoscope get_ang_attribute -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 get_ang_attribute
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on get_ang_attribute
nbop customize_particle_orientation add get_ang_attribute
opexprlanguage -s hscript get_ang_attribute
opuserdata -n '___Version___' -v '' get_ang_attribute

# Node quaternion1 (Vop/quaternion)
opadd -e -n quaternion quaternion1
oplocate -x -19.956099999999999 -y 8.8849599999999995 quaternion1
opspareds "" quaternion1
opparm quaternion1 angle ( 0 ) axis ( 0 1 0 )
chlock quaternion1 -*
chautoscope quaternion1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 quaternion1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on quaternion1
opwire -n -o 1 get_ang_attribute -0 quaternion1
nbop customize_particle_orientation add quaternion1
opexprlanguage -s hscript quaternion1
opuserdata -n '___Version___' -v '' quaternion1

# Node qrotate1 (Vop/qrotate)
opadd -e -n qrotate qrotate1
oplocate -x -18.127800000000001 -y 7.8735099999999996 qrotate1
opspareds "" qrotate1
opparm qrotate1
chlock qrotate1 -*
chautoscope qrotate1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 qrotate1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on qrotate1
opwire -n quaternion1 -0 qrotate1
opwire -n subtract1 -1 qrotate1
nbop customize_particle_orientation add qrotate1
opexprlanguage -s hscript qrotate1
opuserdata -n '___Version___' -v '' qrotate1

# Node subtract1 (Vop/subtract)
opadd -e -n subtract subtract1
oplocate -x -20.370999999999999 -y 6.2788000000000004 subtract1
opspareds "" subtract1
opparm -V 14.0.254 subtract1
chlock subtract1 -*
chautoscope subtract1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 subtract1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on subtract1
opwire -n voxel_pos -0 subtract1
opwire -n particle_center -1 subtract1
nbop customize_particle_orientation add subtract1
opexprlanguage -s hscript subtract1
opuserdata -n '___Version___' -v '14.0.254' subtract1

# Node P (Vop/add)
opadd -e -n add P
oplocate -x 2.5126200000000001 -y 9.4553399999999996 P
opspareds "" P
opparm -V 14.0.254 P
chlock P -*
chautoscope P -*
opcolor -c 0.40000000596046448 1 1 P
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on P
opwire -n rotation3 -0 P
opwire -n particle_center -1 P
opexprlanguage -s hscript P
opuserdata -n '___Version___' -v '14.0.254' P

# Node get_orient_attribute (Vop/getattrib)
opadd -e -n getattrib get_orient_attribute
oplocate -x -17.203199999999999 -y 10.963200000000001 get_orient_attribute
opspareds "" get_orient_attribute
opparm get_orient_attribute signature ( v4 ) opinput ( opinput:0 ) file ( '$HH/geo/defgeo.bgeo' ) atype ( point ) attrib ( orient ) i1 ( 0 ) i2 ( 0 )
chlock get_orient_attribute -*
chautoscope get_orient_attribute -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 get_orient_attribute
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on get_orient_attribute
nbop customize_particle_orientation add get_orient_attribute
opexprlanguage -s hscript get_orient_attribute
opuserdata -n '___Version___' -v '' get_orient_attribute

# Node qrotate2 (Vop/qrotate)
opadd -e -n qrotate qrotate2
oplocate -x -13.962899999999999 -y 9.2993799999999993 qrotate2
opspareds "" qrotate2
opparm qrotate2
chlock qrotate2 -*
chautoscope qrotate2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 qrotate2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on qrotate2
opwire -n -o 1 get_orient_attribute -0 qrotate2
opwire -n rotation1 -1 qrotate2
nbop customize_particle_orientation add qrotate2
opexprlanguage -s hscript qrotate2
opuserdata -n '___Version___' -v '' qrotate2

# Node get_rot_attribute (Vop/getattrib)
opadd -e -n getattrib get_rot_attribute
oplocate -x -11.8316 -y 12.465400000000001 get_rot_attribute
opspareds "" get_rot_attribute
opparm get_rot_attribute signature ( v4 ) opinput ( opinput:0 ) file ( '$HH/geo/defgeo.bgeo' ) atype ( point ) attrib ( rot ) i1 ( 0 ) i2 ( 0 )
chlock get_rot_attribute -*
chautoscope get_rot_attribute -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 get_rot_attribute
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on get_rot_attribute
nbop customize_particle_orientation add get_rot_attribute
opexprlanguage -s hscript get_rot_attribute
opuserdata -n '___Version___' -v '' get_rot_attribute

# Node qrotate3 (Vop/qrotate)
opadd -e -n qrotate qrotate3
oplocate -x -9.5787899999999997 -y 11.433400000000001 qrotate3
opspareds "" qrotate3
opparm qrotate3
chlock qrotate3 -*
chautoscope qrotate3 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 qrotate3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on qrotate3
opwire -n -o 1 get_rot_attribute -0 qrotate3
opwire -n rotation2 -1 qrotate3
nbop customize_particle_orientation add qrotate3
opexprlanguage -s hscript qrotate3
opuserdata -n '___Version___' -v '' qrotate3

# Node rotation1 (Vop/twoway)
opadd -e -n twoway rotation1
oplocate -x -16.0029 -y 8.0918399999999995 rotation1
opspareds "" rotation1
opparm rotation1 signature ( v ) condtype ( 0 ) input2 ( 0 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock rotation1 -*
chautoscope rotation1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 rotation1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on rotation1
opwire -n get_ang_attribute -0 rotation1
opwire -n qrotate1 -1 rotation1
opwire -n subtract1 -2 rotation1
nbop customize_particle_orientation add rotation1
opexprlanguage -s hscript rotation1
opuserdata -n '___Version___' -v '' rotation1

# Node rotation2 (Vop/twoway)
opadd -e -n twoway rotation2
oplocate -x -11.4961 -y 9.60459 rotation2
opspareds "" rotation2
opparm rotation2 signature ( v ) condtype ( 0 ) input2 ( 0 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock rotation2 -*
chautoscope rotation2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 rotation2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on rotation2
opwire -n get_orient_attribute -0 rotation2
opwire -n qrotate2 -1 rotation2
opwire -n rotation1 -2 rotation2
nbop customize_particle_orientation add rotation2
opexprlanguage -s hscript rotation2
opuserdata -n '___Version___' -v '' rotation2

# Node rotation3 (Vop/twoway)
opadd -e -n twoway rotation3
oplocate -x -6.5451800000000002 -y 11.292199999999999 rotation3
opspareds "" rotation3
opparm rotation3 signature ( v ) condtype ( 0 ) input2 ( 0 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock rotation3 -*
chautoscope rotation3 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 rotation3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on rotation3
opwire -n get_rot_attribute -0 rotation3
opwire -n qrotate3 -1 rotation3
opwire -n rotation2 -2 rotation3
nbop customize_particle_orientation add rotation3
opexprlanguage -s hscript rotation3
opuserdata -n '___Version___' -v '' rotation3

# Node CP (Vop/null)
opadd -e -n null CP
oplocate -x -4.1392699999999998 -y 9.0598500000000008 CP
opspareds "" CP
opparm CP  outputnum ( 1 )
opparm -V 14.0.254 CP outputnum ( 1 ) outputname1 ( "" )
chlock CP -*
chautoscope CP -*
opcolor -c 0.40000000596046448 1 1 CP
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on CP
opwire -n rotation3 -0 CP
opexprlanguage -s hscript CP
opuserdata -n '___Version___' -v '14.0.254' CP
opcf ..

# Node divide1 (Vop/divide)
opadd -e -n divide divide1
oplocate -x -21.139087677001953 -y 8.2904300689697266 divide1
opspareds "" divide1
opparm -V 14.0.254 divide1
chlock divide1 -*
chautoscope divide1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 divide1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on divide1
opwire -n switch1 -0 divide1
opwire -n -o 5 Inputs -1 divide1
opexprlanguage -s hscript divide1
opuserdata -n '___Version___' -v '14.0.254' divide1

# Node direction (Vop/normalize)
opadd -e -n normalize direction
oplocate -x -17.750411987304688 -y 7.5322451591491699 direction
opspareds "" direction
opparm direction signature ( default ) vec ( 1 0 0 ) vec_p ( 1 0 0 ) vec_v ( 1 0 0 ) vec_v4 ( 0 0 0 1 ) vec_un ( 1 0 0 ) vec_up ( 1 0 0 ) vec_uv ( 1 0 0 )
chlock direction -*
chautoscope direction -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 direction
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on direction
opwire -n divide1 -0 direction
opexprlanguage -s hscript direction
opuserdata -n '___Version___' -v '' direction

# Node 3d_noise (Vop/aanoise)
opadd -e -n aanoise 3d_noise
oplocate -x -8.0630699999999997 -y 6.6418600000000003 3d_noise
opspareds "" 3d_noise
opparm 3d_noise signature ( default ) pos_ff ( 0 ) pos ( 0 0 0 ) pos_fp ( 0 0 0 0 ) freq_ff ( 1 ) freq ( 1 1 1 ) freq_fp ( 1 1 1 1 ) offset_ff ( 0 ) offset ( 0 0 0 ) offset_fp ( 0 0 0 0 ) amp ( 1 ) rough ( 0.5 ) maxoctave ( 8 ) noisetype ( noise )
chlock 3d_noise -*
chautoscope 3d_noise -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 3d_noise
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 3d_noise
opwire -n 3d_pos -0 3d_noise
opwire -n 3d_frequency -1 3d_noise
opwire -n Offset -2 3d_noise
opwire -n Amplitude -3 3d_noise
opwire -n Roughness -4 3d_noise
opwire -n Ocataves -5 3d_noise
opexprlanguage -s hscript 3d_noise
opuserdata -n '___Version___' -v '' 3d_noise

# Node abs1 (Vop/abs)
opadd -e -n abs abs1
oplocate -x -1.1029899999999999 -y 7.3573599999999999 abs1
opspareds "" abs1
opparm abs1 signature ( default ) val ( 1 ) val_i ( 1 ) val_v ( 1 1 1 ) val_p ( 1 1 1 ) val_n ( 1 1 1 ) val_c ( 1 1 1 ) val_v4 ( 1 1 1 1 ) val_uf ( 1 ) val_uv ( 1 1 1 ) val_up ( 1 1 1 ) val_un ( 1 1 1 ) val_uc ( 1 1 1 )
chlock abs1 -*
chautoscope abs1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 abs1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on abs1
opwire -n noise_val -0 abs1
opexprlanguage -s hscript abs1
opuserdata -n '___Version___' -v '' abs1

# Node pow1 (Vop/pow)
opadd -e -n pow pow1
oplocate -x 6.3232299999999997 -y 6.5202600000000004 pow1
opspareds "" pow1
opparm pow1 signature ( default ) val ( 1 ) val_v ( 1 1 1 ) val_p ( 1 1 1 ) val_n ( 1 1 1 ) val_c ( 1 1 1 ) val_v4 ( 1 1 1 1 ) val_uf ( 1 ) val_uv ( 1 1 1 ) val_up ( 1 1 1 ) val_un ( 1 1 1 ) val_uc ( 1 1 1 ) exp ( 1 )
chlock pow1 -*
chautoscope pow1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 pow1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on pow1
opwire -n noise_value -0 pow1
opwire -n Excavation -1 pow1
opexprlanguage -s hscript pow1
opuserdata -n '___Version___' -v '' pow1

# Node length1 (Vop/length)
opadd -e -n length length1
oplocate -x -11.9124 -y 9.9906500000000005 length1
opspareds "" length1
opparm length1 signature ( default ) vec ( 1 1 1 ) vec_p ( 1 1 1 ) vec_n ( 1 1 1 ) vec_v4 ( 1 1 1 1 ) vec_uv ( 1 1 1 ) vec_up ( 1 1 1 ) vec_un ( 1 1 1 )
chlock length1 -*
chautoscope length1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 length1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on length1
opwire -n divide1 -0 length1
opexprlanguage -s hscript length1
opuserdata -n '___Version___' -v '' length1

# Node subtract1 (Vop/subtract)
opadd -e -n subtract subtract1
oplocate -x 9.4113799999999994 -y 8.5167400000000004 subtract1
opspareds "" subtract1
opparm -V 14.0.254 subtract1
chlock subtract1 -*
chautoscope subtract1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 subtract1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on subtract1
opwire -n length1 -0 subtract1
opwire -n pow1 -1 subtract1
opexprlanguage -s hscript subtract1
opuserdata -n '___Version___' -v '14.0.254' subtract1

# Node clamp1 (Vop/clamp)
opadd -e -n clamp clamp1
oplocate -x 16.9497 -y 5.8520500000000002 clamp1
opspareds "" clamp1
opparm clamp1 signature ( default ) min ( 0 ) max ( 1 ) min_i ( 0 ) max_i ( 1 ) min_v ( 0 0 0 ) max_v ( 1 1 1 ) min_p ( 0 0 0 ) max_p ( 1 1 1 ) min_n ( 0 0 0 ) max_n ( 1 1 1 ) min_c ( 0 0 0 ) max_c ( 1 1 1 ) min_v4 ( 0 0 0 0 ) max_v4 ( 1 1 1 1 ) min_uf ( 0 ) max_uf ( 1 ) min_uv ( 0 0 0 ) max_uv ( 1 1 1 ) min_up ( 0 0 0 ) max_up ( 1 1 1 ) min_un ( 0 0 0 ) max_un ( 1 1 1 ) min_uc ( 0 0 0 ) max_uc ( 1 1 1 )
chlock clamp1 -*
chautoscope clamp1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 clamp1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on clamp1
opwire -n fit1 -0 clamp1
opexprlanguage -s hscript clamp1
opuserdata -n '___Version___' -v '' clamp1

# Node Excavation (Vop/parameter)
opadd -e -n parameter Excavation
oplocate -x 3.9055900000000001 -y 5.2791499999999996 Excavation
opspareds "" Excavation
opparm -V 14.0.254 Excavation parmscope ( shaderparm ) parmaccess ( "" ) parmname ( excavation ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Excavation ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.070000000000000007 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Excavation -*
chautoscope Excavation -*
opcolor -c 1 1 0.40000000596046448 Excavation
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Excavation
opexprlanguage -s hscript Excavation
opuserdata -n '___Version___' -v '14.0.254' Excavation

# Node fit1 (Vop/fit)
opadd -e -n fit fit1
oplocate -x 14.521599999999999 -y 5.5696500000000002 fit1
opspareds "" fit1
opparm fit1 signature ( default ) srcmin ( 0 ) srcmax ( 1 ) destmin ( 1 ) destmax ( 0 ) srcmin_v ( 0 0 0 ) srcmax_v ( 1 1 1 ) destmin_v ( 0 0 0 ) destmax_v ( 1 1 1 ) srcmin_p ( 0 0 0 ) srcmax_p ( 1 1 1 ) destmin_p ( 0 0 0 ) destmax_p ( 1 1 1 ) srcmin_n ( 0 0 0 ) srcmax_n ( 1 1 1 ) destmin_n ( 0 0 0 ) destmax_n ( 1 1 1 ) srcmin_c ( 0 0 0 ) srcmax_c ( 1 1 1 ) destmin_c ( 0 0 0 ) destmax_c ( 1 1 1 ) srcmin_v4 ( 0 0 0 0 ) srcmax_v4 ( 1 1 1 1 ) destmin_v4 ( 0 0 0 0 ) destmax_v4 ( 1 1 1 1 ) srcmin_uf ( 0 ) srcmax_uf ( 1 ) destmin_uf ( 0 ) destmax_uf ( 1 ) srcmin_uv ( 0 0 0 ) srcmax_uv ( 1 1 1 ) destmin_uv ( 0 0 0 ) destmax_uv ( 1 1 1 ) srcmin_up ( 0 0 0 ) srcmax_up ( 1 1 1 ) destmin_up ( 0 0 0 ) destmax_up ( 1 1 1 ) srcmin_un ( 0 0 0 ) srcmax_un ( 1 1 1 ) destmin_un ( 0 0 0 ) destmax_un ( 1 1 1 ) srcmin_uc ( 0 0 0 ) srcmax_uc ( 1 1 1 ) destmin_uc ( 0 0 0 ) destmax_uc ( 1 1 1 )
chlock fit1 -*
chautoscope fit1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 fit1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on fit1
opwire -n subtract1 -0 fit1
opwire -n negate_falloff -1 fit1
opwire -n scale_falloff -2 fit1
opexprlanguage -s hscript fit1
opuserdata -n '___Version___' -v '' fit1

# Node Falloff (Vop/parameter)
opadd -e -n parameter Falloff
oplocate -x 4.3218800000000002 -y 4.2399199999999997 Falloff
opspareds "" Falloff
opparm -V 14.0.254 Falloff parmscope ( shaderparm ) parmaccess ( "" ) parmname ( falloff ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Falloff ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.10000000000000001 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Falloff -*
chautoscope Falloff -*
opcolor -c 1 1 0.40000000596046448 Falloff
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Falloff
opexprlanguage -s hscript Falloff
opuserdata -n '___Version___' -v '14.0.254' Falloff

# Node scale_falloff (Vop/mulconst)
opadd -e -n mulconst scale_falloff
oplocate -x 8.4952400000000008 -y 5.3601099999999997 scale_falloff
opspareds "" scale_falloff
opparm scale_falloff signature ( default ) mulconst ( 0.5 )
chlock scale_falloff -*
chautoscope scale_falloff -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 scale_falloff
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on scale_falloff
opwire -n min_clamp_falloff -0 scale_falloff
opexprlanguage -s hscript scale_falloff
opuserdata -n '___Version___' -v '' scale_falloff

# Node negate_falloff (Vop/negate)
opadd -e -n negate negate_falloff
oplocate -x 10.721299999999999 -y 6.4376600000000002 negate_falloff
opspareds "" negate_falloff
opparm negate_falloff signature ( f )
chlock negate_falloff -*
chautoscope negate_falloff -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 negate_falloff
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on negate_falloff
opwire -n scale_falloff -0 negate_falloff
opexprlanguage -s hscript negate_falloff
opuserdata -n '___Version___' -v '' negate_falloff

# Node min_clamp_falloff (Vop/clamp)
opadd -e -n clamp min_clamp_falloff
oplocate -x 6.7704199999999997 -y 4.4910100000000002 min_clamp_falloff
opspareds "" min_clamp_falloff
opparm min_clamp_falloff signature ( default ) min ( 0.0001 ) max ( 1 ) min_i ( 0 ) max_i ( 1 ) min_v ( 0 0 0 ) max_v ( 1 1 1 ) min_p ( 0 0 0 ) max_p ( 1 1 1 ) min_n ( 0 0 0 ) max_n ( 1 1 1 ) min_c ( 0 0 0 ) max_c ( 1 1 1 ) min_v4 ( 0 0 0 0 ) max_v4 ( 1 1 1 1 ) min_uf ( 0 ) max_uf ( 1 ) min_uv ( 0 0 0 ) max_uv ( 1 1 1 ) min_up ( 0 0 0 ) max_up ( 1 1 1 ) min_un ( 0 0 0 ) max_un ( 1 1 1 ) min_uc ( 0 0 0 ) max_uc ( 1 1 1 )
chlock min_clamp_falloff -*
chautoscope min_clamp_falloff -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 min_clamp_falloff
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on min_clamp_falloff
opwire -n Falloff -0 min_clamp_falloff
opwire -n Falloff -2 min_clamp_falloff
opexprlanguage -s hscript min_clamp_falloff
opuserdata -n '___Version___' -v '' min_clamp_falloff

# Node complement1 (Vop/complement)
opadd -e -n complement complement1
oplocate -x 0.62414199999999997 -y 6.7391300000000003 complement1
opspareds "" complement1
opparm complement1 signature ( default ) val ( 1 ) val_i ( 1 ) val_v ( 1 1 1 ) val_p ( 1 1 1 ) val_n ( 1 1 1 ) val_c ( 1 1 1 ) val_v4 ( 1 1 1 1 ) val_uf ( 1 ) val_uv ( 1 1 1 ) val_up ( 1 1 1 ) val_un ( 1 1 1 ) val_uc ( 1 1 1 )
chlock complement1 -*
chautoscope complement1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 complement1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on complement1
opwire -n abs1 -0 complement1
opexprlanguage -s hscript complement1
opuserdata -n '___Version___' -v '' complement1

# Node clamp2 (Vop/clamp)
opadd -e -n clamp clamp2
oplocate -x 1.6280600000000001 -y 5.7804500000000001 clamp2
opspareds "" clamp2
opparm clamp2 signature ( default ) min ( 0 ) max ( 1 ) min_i ( 0 ) max_i ( 1 ) min_v ( 0 0 0 ) max_v ( 1 1 1 ) min_p ( 0 0 0 ) max_p ( 1 1 1 ) min_n ( 0 0 0 ) max_n ( 1 1 1 ) min_c ( 0 0 0 ) max_c ( 1 1 1 ) min_v4 ( 0 0 0 0 ) max_v4 ( 1 1 1 1 ) min_uf ( 0 ) max_uf ( 1 ) min_uv ( 0 0 0 ) max_uv ( 1 1 1 ) min_up ( 0 0 0 ) max_up ( 1 1 1 ) min_un ( 0 0 0 ) max_un ( 1 1 1 ) min_uc ( 0 0 0 ) max_uc ( 1 1 1 )
chlock clamp2 -*
chautoscope clamp2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 clamp2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on clamp2
opwire -n noise_val -0 clamp2
opwire -n noise_val -2 clamp2
opexprlanguage -s hscript clamp2
opuserdata -n '___Version___' -v '' clamp2

# Node noise_value (Vop/switch)
opadd -e -n switch noise_value
oplocate -x 3.8746700000000001 -y 7.1538199999999996 noise_value
opspareds "" noise_value
opparm -V 14.0.254 noise_value switcher ( 0 ) outofbounds ( last )
chlock noise_value -*
chautoscope noise_value -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 noise_value
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on noise_value
opwire -n Noise_Treatment -0 noise_value
opwire -n abs1 -1 noise_value
opwire -n complement1 -2 noise_value
opwire -n clamp2 -3 noise_value
opexprlanguage -s hscript noise_value
opuserdata -n '___Version___' -v '14.0.254' noise_value

# Node Noise_Treatment (Vop/parameter)
opadd -e -n parameter Noise_Treatment
oplocate -x 0.43536200000000003 -y 8.8107299999999995 Noise_Treatment
opspareds "" Noise_Treatment
opparm -V 14.0.254 Noise_Treatment parmscope ( shaderparm ) parmaccess ( "" ) parmname ( treatment ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Noise Treatment' ) showlabel ( on ) parmtype ( int ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( on ) menuchoices ( '0 "Abs" 1 "1 - Abs" 2 "Clamp"' ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Noise_Treatment -*
chautoscope Noise_Treatment -*
opcolor -c 1 1 0.40000000596046448 Noise_Treatment
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Noise_Treatment
opexprlanguage -s hscript Noise_Treatment
opuserdata -n '___Version___' -v '14.0.254' Noise_Treatment

# Node 3d_pos (Vop/switch)
opadd -e -n switch 3d_pos
oplocate -x -15.4884 -y 8.0080299999999998 3d_pos
opspareds "" 3d_pos
opparm -V 14.0.254 3d_pos switcher ( 0 ) outofbounds ( last )
chlock 3d_pos -*
chautoscope 3d_pos -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 3d_pos
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 3d_pos
opwire -n Noise_Lookup -0 3d_pos
opwire -n direction -1 3d_pos
opwire -n add2 -2 3d_pos
opwire -n switch1 -3 3d_pos
opwire -n divide1 -4 3d_pos
opexprlanguage -s hscript 3d_pos
opuserdata -n '___Version___' -v '14.0.254' 3d_pos

# Node Noise_Lookup (Vop/parameter)
opadd -e -n parameter Noise_Lookup
oplocate -x -18.146059036254883 -y 10.654242515563965 Noise_Lookup
opspareds "" Noise_Lookup
opparm -V 14.0.254 Noise_Lookup parmscope ( shaderparm ) parmaccess ( "" ) parmname ( lookup ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Noise Lookup' ) showlabel ( on ) parmtype ( int ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( on ) menuchoices ( '0 "Displacement" 1 "World Space" 2 "Local Space" 3 "Unit Space"' ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Noise_Lookup -*
chautoscope Noise_Lookup -*
opcolor -c 1 1 0.40000000596046448 Noise_Lookup
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Noise_Lookup
opexprlanguage -s hscript Noise_Lookup
opuserdata -n '___Version___' -v '14.0.254' Noise_Lookup

# Node Frequency (Vop/parameter)
opadd -e -n parameter Frequency
oplocate -x -17.312799999999999 -y 2.9494899999999999 Frequency
opspareds "" Frequency
opparm -V 14.0.254 Frequency parmscope ( shaderparm ) parmaccess ( "" ) parmname ( freq ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Frequency ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 1.74 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Frequency -*
chautoscope Frequency -*
opcolor -c 1 1 0.40000000596046448 Frequency
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Frequency
opexprlanguage -s hscript Frequency
opuserdata -n '___Version___' -v '14.0.254' Frequency

# Node 3d_frequency (Vop/floattovec)
opadd -e -n floattovec 3d_frequency
oplocate -x -15.3462 -y 4.1244899999999998 3d_frequency
opspareds "" 3d_frequency
opparm 3d_frequency signature ( default ) fval1 ( 0 ) fval2 ( 0 ) fval3 ( 0 ) fval1_uv ( 0 ) fval2_uv ( 0 ) fval3_uv ( 0 )
chlock 3d_frequency -*
chautoscope 3d_frequency -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 3d_frequency
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 3d_frequency
opwire -n Frequency -0 3d_frequency
opwire -n Frequency -1 3d_frequency
opwire -n Frequency -2 3d_frequency
opexprlanguage -s hscript 3d_frequency
opuserdata -n '___Version___' -v '' 3d_frequency

# Node Roughness (Vop/parameter)
opadd -e -n parameter Roughness
oplocate -x -13.04542350769043 -y 0.4955751895904541 Roughness
opspareds "" Roughness
opparm -V 14.0.254 Roughness parmscope ( shaderparm ) parmaccess ( "" ) parmname ( rough ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Roughness ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.5 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Roughness -*
chautoscope Roughness -*
opcolor -c 1 1 0.40000000596046448 Roughness
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Roughness
opexprlanguage -s hscript Roughness
opuserdata -n '___Version___' -v '14.0.254' Roughness

# Node Ocataves (Vop/parameter)
opadd -e -n parameter Ocataves
oplocate -x -11.501758575439453 -y -0.22595334053039551 Ocataves
opspareds "" Ocataves
opparm -V 14.0.254 Ocataves parmscope ( shaderparm ) parmaccess ( "" ) parmname ( ocataves ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Ocataves ) showlabel ( on ) parmtype ( int ) parmtypename ( "" ) floatdef ( 1.74 ) intdef ( 2 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Ocataves -*
chautoscope Ocataves -*
opcolor -c 1 1 0.40000000596046448 Ocataves
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Ocataves
opexprlanguage -s hscript Ocataves
opuserdata -n '___Version___' -v '14.0.254' Ocataves

# Node Offset (Vop/parameter)
opadd -e -n parameter Offset
oplocate -x -15.4032 -y 6.5237400000000001 Offset
opspareds "" Offset
opparm -V 14.0.254 Offset parmscope ( shaderparm ) parmaccess ( "" ) parmname ( offset ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Offset ) showlabel ( on ) parmtype ( vector ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Offset -*
chautoscope Offset -*
opcolor -c 1 1 0.40000000596046448 Offset
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Offset
opexprlanguage -s hscript Offset
opuserdata -n '___Version___' -v '14.0.254' Offset

# Node Billowing_Speed (Vop/parameter)
opadd -e -n parameter Billowing_Speed
oplocate -x -34.326599999999999 -y 5.0532000000000004 Billowing_Speed
opspareds "" Billowing_Speed
opparm -V 14.0.254 Billowing_Speed parmscope ( shaderparm ) parmaccess ( "" ) parmname ( billowingspeed ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Billowing Speed' ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.10000000000000001 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ billowing != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Billowing_Speed -*
chautoscope Billowing_Speed -*
opcolor -c 0.80000001192092896 1 0.80000001192092896 Billowing_Speed
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Billowing_Speed
nbop apply_billowing_motion add Billowing_Speed
opexprlanguage -s hscript Billowing_Speed
opuserdata -n '___Version___' -v '14.0.254' Billowing_Speed

# Node multiply1 (Vop/multiply)
opadd -e -n multiply multiply1
oplocate -x -30.5688 -y 5.6063400000000003 multiply1
opspareds "" multiply1
opparm -V 14.0.254 multiply1
chlock multiply1 -*
chautoscope multiply1 -*
opcolor -c 0.80000001192092896 1 0.80000001192092896 multiply1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply1
opwire -n -o 8 Inputs -0 multiply1
opwire -n Billowing_Speed -1 multiply1
nbop apply_billowing_motion add multiply1
opexprlanguage -s hscript multiply1
opuserdata -n '___Version___' -v '14.0.254' multiply1

# Node switch1 (Vop/switch)
opadd -e -n switch switch1
oplocate -x -23.620699999999999 -y 10.029199999999999 switch1
opspareds "" switch1
opparm -V 14.0.254 switch1 switcher ( 0 ) outofbounds ( last )
chlock switch1 -*
chautoscope switch1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 switch1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on switch1
opwire -n Billowing_Motion -0 switch1
opwire -n -o 1 Inputs -1 switch1
opwire -n frompolar1 -2 switch1
opexprlanguage -s hscript switch1
opuserdata -n '___Version___' -v '14.0.254' switch1

# Node vectofloat1 (Vop/vectofloat)
opadd -e -n vectofloat vectofloat1
oplocate -x -34.566800000000001 -y 8.9270300000000002 vectofloat1
opspareds "" vectofloat1
opparm vectofloat1 signature ( default ) vec ( 0 0 0 ) vec_p ( 0 0 0 ) vec_n ( 0 0 0 ) vec_c ( 0 0 0 ) vec_uv ( 0 0 0 ) vec_up ( 0 0 0 ) vec_un ( 0 0 0 ) vec_uc ( 0 0 0 )
chlock vectofloat1 -*
chautoscope vectofloat1 -*
opcolor -c 0.80000001192092896 1 0.80000001192092896 vectofloat1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on vectofloat1
opwire -n -o 2 Inputs -0 vectofloat1
nbop apply_billowing_motion add vectofloat1
opexprlanguage -s hscript vectofloat1
opuserdata -n '___Version___' -v '' vectofloat1

# Node normalize_theta (Vop/divconst)
opadd -e -n divconst normalize_theta
oplocate -x -30.5688 -y 8.0289300000000008 normalize_theta
opspareds "" normalize_theta
chblockbegin
chadd -t 0 0 normalize_theta divconst
chkey -t 0 -v 1 -m 0 -a 0 -A 0 -T a  -F '2*$PI' normalize_theta/divconst
chblockend
opparm normalize_theta signature ( default ) divconst ( divconst )
chlock normalize_theta -*
chautoscope normalize_theta -*
opcolor -c 0.80000001192092896 1 0.80000001192092896 normalize_theta
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on normalize_theta
opwire -n -o 1 vectofloat1 -0 normalize_theta
nbop apply_billowing_motion add normalize_theta
opexprlanguage -s hscript normalize_theta
opuserdata -n '___Version___' -v '' normalize_theta

# Node normalize_phi (Vop/divconst)
opadd -e -n divconst normalize_phi
oplocate -x -30.5688 -y 7.08399 normalize_phi
opspareds "" normalize_phi
chblockbegin
chadd -t 0 0 normalize_phi divconst
chkey -t 0 -v 1 -m 0 -a 0 -A 0 -T a  -F '$PI' normalize_phi/divconst
chblockend
opparm normalize_phi signature ( default ) divconst ( divconst )
chlock normalize_phi -*
chautoscope normalize_phi -*
opcolor -c 0.80000001192092896 1 0.80000001192092896 normalize_phi
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on normalize_phi
opwire -n -o 2 vectofloat1 -0 normalize_phi
nbop apply_billowing_motion add normalize_phi
opexprlanguage -s hscript normalize_phi
opuserdata -n '___Version___' -v '' normalize_phi

# Node add1 (Vop/add)
opadd -e -n add add1
oplocate -x -28.5136 -y 6.80159 add1
opspareds "" add1
opparm -V 14.0.254 add1
chlock add1 -*
chautoscope add1 -*
opcolor -c 0.80000001192092896 1 0.80000001192092896 add1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on add1
opwire -n normalize_phi -0 add1
opwire -n multiply1 -1 add1
nbop apply_billowing_motion add add1
opexprlanguage -s hscript add1
opuserdata -n '___Version___' -v '14.0.254' add1

# Node frompolar1 (Vop/frompolar)
opadd -e -n frompolar frompolar1
oplocate -x -26.665199999999999 -y 8.0635200000000005 frompolar1
opspareds "" frompolar1
opparm frompolar1 u ( 0 ) v ( 0 ) radius ( 1 ) ispace ( unit )
chlock frompolar1 -*
chautoscope frompolar1 -*
opcolor -c 0.80000001192092896 1 0.80000001192092896 frompolar1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on frompolar1
opwire -n normalize_theta -0 frompolar1
opwire -n add1 -1 frompolar1
opwire -n vectofloat1 -2 frompolar1
nbop apply_billowing_motion add frompolar1
opexprlanguage -s hscript frompolar1
opuserdata -n '___Version___' -v '' frompolar1

# Node 4d_offset (Vop/vectohvec)
opadd -e -n vectohvec 4d_offset
oplocate -x -13.340299999999999 -y 5.6186199999999999 4d_offset
opspareds "" 4d_offset
opparm 4d_offset vec ( 0 0 0 ) fval4 ( 0 )
chlock 4d_offset -*
chautoscope 4d_offset -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 4d_offset
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 4d_offset
opwire -n Offset -0 4d_offset
opwire -n multiply2 -1 4d_offset
opexprlanguage -s hscript 4d_offset
opuserdata -n '___Version___' -v '' 4d_offset

# Node 4d_frequency (Vop/floattohvec)
opadd -e -n floattohvec 4d_frequency
oplocate -x -13.333500000000001 -y 3.08866 4d_frequency
opspareds "" 4d_frequency
opparm 4d_frequency fval1 ( 0 ) fval2 ( 0 ) fval3 ( 0 ) fval4 ( 0 )
chlock 4d_frequency -*
chautoscope 4d_frequency -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 4d_frequency
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 4d_frequency
opwire -n Frequency -0 4d_frequency
opwire -n Frequency -1 4d_frequency
opwire -n Frequency -2 4d_frequency
opwire -n Frequency -3 4d_frequency
opexprlanguage -s hscript 4d_frequency
opuserdata -n '___Version___' -v '' 4d_frequency

# Node 4d_pos (Vop/vectohvec)
opadd -e -n vectohvec 4d_pos
oplocate -x -13.262 -y 7.3478599999999998 4d_pos
opspareds "" 4d_pos
opparm 4d_pos vec ( 0 0 0 ) fval4 ( 0 )
chlock 4d_pos -*
chautoscope 4d_pos -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 4d_pos
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 4d_pos
opwire -n 3d_pos -0 4d_pos
opexprlanguage -s hscript 4d_pos
opuserdata -n '___Version___' -v '' 4d_pos

# Node Billowing_Motion (Vop/parameter)
opadd -e -n parameter Billowing_Motion
oplocate -x -26.450500000000002 -y 12.151199999999999 Billowing_Motion
opspareds "" Billowing_Motion
opparm -V 14.0.254 Billowing_Motion parmscope ( shaderparm ) parmaccess ( "" ) parmname ( billowing ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Billowing Motion' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Billowing_Motion -*
chautoscope Billowing_Motion -*
opcolor -c 1 1 0.40000000596046448 Billowing_Motion
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Billowing_Motion
opexprlanguage -s hscript Billowing_Motion
opuserdata -n '___Version___' -v '14.0.254' Billowing_Motion

# Node Swirling_Motion (Vop/parameter)
opadd -e -n parameter Swirling_Motion
oplocate -x -6.8648800000000003 -y 8.6094399999999993 Swirling_Motion
opspareds "" Swirling_Motion
opparm -V 14.0.254 Swirling_Motion parmscope ( shaderparm ) parmaccess ( "" ) parmname ( swirling ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Swirling Motion' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Swirling_Motion -*
chautoscope Swirling_Motion -*
opcolor -c 1 1 0.40000000596046448 Swirling_Motion
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Swirling_Motion
opexprlanguage -s hscript Swirling_Motion
opuserdata -n '___Version___' -v '14.0.254' Swirling_Motion

# Node Swirling_Speed (Vop/parameter)
opadd -e -n parameter Swirling_Speed
oplocate -x -22.192499999999999 -y 4.6841600000000003 Swirling_Speed
opspareds "" Swirling_Speed
opparm -V 14.0.254 Swirling_Speed parmscope ( shaderparm ) parmaccess ( "" ) parmname ( swirlingspeed ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Swirling Speed' ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.10000000000000001 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ billowing != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Swirling_Speed -*
chautoscope Swirling_Speed -*
opcolor -c 1 1 0.40000000596046448 Swirling_Speed
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Swirling_Speed
opexprlanguage -s hscript Swirling_Speed
opuserdata -n '___Version___' -v '14.0.254' Swirling_Speed

# Node multiply2 (Vop/multiply)
opadd -e -n multiply multiply2
oplocate -x -18.537099999999999 -y 5.6063400000000003 multiply2
opspareds "" multiply2
opparm -V 14.0.254 multiply2
chlock multiply2 -*
chautoscope multiply2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply2
opwire -n -o 8 Inputs -0 multiply2
opwire -n Swirling_Speed -1 multiply2
opexprlanguage -s hscript multiply2
opuserdata -n '___Version___' -v '14.0.254' multiply2

# Node 4d_noise (Vop/aanoise)
opadd -e -n aanoise 4d_noise
oplocate -x -8.0630699999999997 -y 4.3787700000000003 4d_noise
opspareds "" 4d_noise
opparm 4d_noise signature ( fp ) pos_ff ( 0 ) pos ( 0 0 0 ) pos_fp ( 0 0 0 0 ) freq_ff ( 1 ) freq ( 1 1 1 ) freq_fp ( 1 1 1 1 ) offset_ff ( 0 ) offset ( 0 0 0 ) offset_fp ( 0 0 0 0 ) amp ( 1 ) rough ( 0.5 ) maxoctave ( 8 ) noisetype ( noise )
chlock 4d_noise -*
chautoscope 4d_noise -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 4d_noise
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 4d_noise
opwire -n 4d_pos -0 4d_noise
opwire -n 4d_frequency -1 4d_noise
opwire -n 4d_offset -2 4d_noise
opwire -n Amplitude -3 4d_noise
opwire -n Roughness -4 4d_noise
opwire -n Ocataves -5 4d_noise
opexprlanguage -s hscript 4d_noise
opuserdata -n '___Version___' -v '' 4d_noise

# Node noise_val (Vop/twoway)
opadd -e -n twoway noise_val
oplocate -x -4.2744999999999997 -y 5.99925 noise_val
opspareds "" noise_val
opparm noise_val signature ( default ) condtype ( 1 ) input2 ( 0 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock noise_val -*
chautoscope noise_val -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 noise_val
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on noise_val
opwire -n Swirling_Motion -0 noise_val
opwire -n 3d_noise -1 noise_val
opwire -n 4d_noise -2 noise_val
opexprlanguage -s hscript noise_val
opuserdata -n '___Version___' -v '' noise_val

# Node add2 (Vop/add)
opadd -e -n add add2
oplocate -x -21.989009857177734 -y 11.118959426879883 add2
opspareds "" add2
opparm -V 14.0.254 add2
chlock add2 -*
chautoscope add2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 add2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on add2
opwire -n -o 4 Inputs -0 add2
opwire -n switch1 -1 add2
opexprlanguage -s hscript add2
opuserdata -n '___Version___' -v '14.0.254' add2

# Node Amplitude (Vop/parameter)
opadd -e -n parameter Amplitude
oplocate -x -14.204265594482422 -y 1.6479486227035522 Amplitude
opspareds "" Amplitude
opparm -V 14.0.254 Amplitude parmscope ( shaderparm ) parmaccess ( "" ) parmname ( amp ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Amplitude ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 1 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( "" ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Amplitude -*
chautoscope Amplitude -*
opcolor -c 1 1 0.40000000596046448 Amplitude
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Amplitude
opexprlanguage -s hscript Amplitude
opuserdata -n '___Version___' -v '14.0.254' Amplitude
opcf Inputs
oporder -e voxel_pos voxel_size particle_center particle_radius particle_index Frame Time TimeInc suboutput1 subinput1 length1 split spherical_coord atan1 divide1 trig1 get_ang_attribute quaternion1 qrotate1 subtract1 P get_orient_attribute qrotate2 get_rot_attribute qrotate3 rotation1 rotation2 rotation3 CP
opcf ..
oporder -e density_scale Inputs divide1 direction 3d_noise abs1 pow1 length1 subtract1 clamp1 Excavation fit1 Falloff scale_falloff negate_falloff min_clamp_falloff complement1 clamp2 noise_value Noise_Treatment 3d_pos Noise_Lookup Frequency 3d_frequency Roughness Ocataves Offset Billowing_Speed multiply1 switch1 vectofloat1 normalize_theta normalize_phi add1 frompolar1 4d_offset 4d_frequency 4d_pos Billowing_Motion Swirling_Motion Swirling_Speed multiply2 4d_noise noise_val add2 Amplitude

opcf ..
