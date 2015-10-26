# This creation script augment the "Rasterize Points" SOP with a default
# VOP subnetwork and UI parameters for cloud and velocity field modeling.
#
# Installation:
# 1. Verify that script file name matches "Rasterize Points" operator name.
# 2. Move to scripts/sop/ subdirectory.
# 3. Update HOUDINI_SCRIPT_PATH environment variable.
#
# See "Using Creation Scripts" section in HDK documentation for more details.

\set noalias = 1
#
#  Creation script for DW_OpenVDBRasterizePoints operator
#

if ( "$arg1" == "" ) then
    echo This script is intended as a creation script
    exit
endif

# Node $arg1 (Sop/DW_OpenVDBRasterizePoints)
opspareds '    parm { 	name	"pointgroup" 	baseparm 	export	none     }     parm { 	name	"voxelsize" 	baseparm 	export	none     }     parm { 	name	"createdensity" 	baseparm 	export	none     }     parm { 	name	"compositing" 	baseparm 	export	none     }     parm { 	name	"densityscale" 	baseparm 	export	none     }     parm { 	name	"particlescale" 	baseparm 	export	none     }     parm { 	name	"solidratio" 	baseparm 	export	none     }     parm { 	name	"attributes" 	baseparm 	export	none     }     parm { 	name	"noiseheading" 	baseparm 	export	none     }     parm { 	name	"modeling" 	baseparm 	export	none     }     group { 	name	"density_folder" 	label	"Density"  	parm { 	    name	"process_density" 	    label	"Process Density" 	    type	toggle 	    default	{ "off" } 	    disablewhen	"{ modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"lookup" 	    label	"Noise Lookup" 	    type	integer 	    default	{ "0" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    menu	{ 		"0"	"Displacement" 		"1"	"World Space" 		"2"	"Local Space" 		"3"	"Unit Space" 	    } 	    range	{ 0 10 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"treatment" 	    label	"Noise Treatment" 	    type	integer 	    default	{ "0" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    menu	{ 		"0"	"Abs" 		"1"	"1 - Abs" 		"2"	"Clamp" 	    } 	    range	{ 0 10 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"amp" 	    label	"Amplitude" 	    type	float 	    default	{ "1" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 2 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"excavation" 	    label	"Excavation" 	    type	float 	    default	{ "0.07" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"falloff" 	    label	"Falloff" 	    type	float 	    default	{ "0.1" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"freq" 	    label	"Frequency" 	    type	float 	    default	{ "1.74" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 10 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"rough" 	    label	"Roughness" 	    type	float 	    default	{ "0.5" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"ocataves" 	    label	"Ocataves" 	    type	integer 	    default	{ "2" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 10 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"offset" 	    label	"Offset" 	    type	float 	    size	3 	    default	{ "0" "0" "0" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 10 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"billowing" 	    label	"Billowing Motion" 	    type	toggle 	    default	{ "off" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"billowingspeed" 	    label	"Billowing Speed" 	    type	float 	    default	{ "0.1" } 	    disablewhen	"{ billowing != 1 } { process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"swirling" 	    label	"Swirling Motion" 	    type	toggle 	    default	{ "off" } 	    disablewhen	"{ process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"swirlingspeed" 	    label	"Swirling Speed" 	    type	float 	    default	{ "0.1" } 	    disablewhen	"{ swirling != 1 } { process_density != 1 } { modeling != 1 } { createdensity != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	}     }      group { 	name	"density_folder_1" 	label	"Velocity"  	parm { 	    name	"process_velocity" 	    label	"Process Velocity" 	    type	toggle 	    default	{ "off" } 	    help	"Requires a velocity point attribute named \'v\'." 	    disablewhen	"{ modeling != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"velocity_scale" 	    label	"Velocity Scale" 	    type	float 	    default	{ "1" } 	    help	"Scale the input velocity" 	    disablewhen	"{ process_velocity != 1 } { modeling != 1 }" 	    range	{ -2 2 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"radial_component" 	    label	"Radial" 	    type	float 	    default	{ "0" } 	    help	"Add velocity components that diverge or converge from the center." 	    disablewhen	"{ process_velocity != 1 } { modeling != 1 }" 	    range	{ -1 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"rotational_component" 	    label	"Rotational" 	    type	float 	    default	{ "0" } 	    help	"Add velocity components that circulate about the input velocity direction." 	    disablewhen	"{ process_velocity != 1 } { modeling != 1 }" 	    range	{ -1 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"orthogonal_component" 	    label	"Orthogonal" 	    type	float 	    default	{ "0" } 	    help	"Add velocity components that are orthogonal to the input velocity direction." 	    disablewhen	"{ process_velocity != 1 } { modeling != 1 }" 	    range	{ -1 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"scale_components" 	    label	"Scale Radial, Rotational and Orthogonal Components " 	    type	toggle 	    default	{ "on" } 	    help	"Scale radial, rotational and orthogonal components with input velocity magnitude." 	    disablewhen	"{ process_velocity != 1 } { modeling != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"velocity_noise" 	    label	"Multiplicative Noise" 	    type	toggle 	    default	{ "off" } 	    help	"Scale final velocity using noise" 	    disablewhen	"{ process_velocity != 1 } { modeling != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"velocity_freq" 	    label	"Frequency" 	    type	float 	    size	3 	    default	{ "0.5" "0.5" "0.5" } 	    help	"Noise frequency" 	    disablewhen	"{ process_velocity != 1 } { velocity_noise != 1 } { modeling != 1 }" 	    range	{ 0 10 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"velocity_amp" 	    label	"Amplitude" 	    type	float 	    default	{ "1" } 	    help	"Noise amplitude" 	    disablewhen	"{ process_velocity != 1 } { velocity_noise != 1 } { modeling != 1 }" 	    range	{ -2 2 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"magnitude_falloff" 	    label	"Magnitude Falloff" 	    type	toggle 	    default	{ "off" } 	    help	"Apply radial velocity magnitude falloff" 	    disablewhen	"{ process_velocity != 1 } { modeling != 1 }" 	    range	{ 0 1 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	} 	parm { 	    name	"velocity_falloff" 	    label	"Falloff" 	    type	ramp_flt 	    default	{ "2" } 	    disablewhen	"{ process_velocity != 1 } { magnitude_falloff != 1 } { modeling != 1 }" 	    range	{ 1! 10 } 	    export	none 	    parmtag	{ "parmvop" "1" } 	    parmtag	{ "rampbasis_var" "velocity_falloff_the_basis_strings" } 	    parmtag	{ "rampbasisdefault" "catmull-rom" } 	    parmtag	{ "rampkeys_var" "velocity_falloff_the_key_positions" } 	    parmtag	{ "rampshowcontrolsdefault" "0" } 	    parmtag	{ "rampvalues_var" "velocity_falloff_the_key_values" } 	    parmtag	{ "shaderparmcontexts" "cvex" } 	}     }  ' $arg1
opparm $arg1  velocity_falloff ( 2 )
opparm -V 14.0.254 $arg1 pointgroup ( "" ) voxelsize ( 0.10000000000000001 ) createdensity ( on ) compositing ( max ) densityscale ( 1 ) particlescale ( 1 ) solidratio ( 0 ) attributes ( "" ) noiseheading ( ) modeling ( off ) density_folder ( 0 0 ) process_density ( off ) lookup ( 0 ) treatment ( 0 ) amp ( 1 ) excavation ( 0.070000000000000007 ) falloff ( 0.10000000000000001 ) freq ( 1.74 ) rough ( 0.5 ) ocataves ( 2 ) offset ( 0 0 0 ) billowing ( off ) billowingspeed ( 0.10000000000000001 ) swirling ( off ) swirlingspeed ( 0.10000000000000001 ) process_velocity ( off ) velocity_scale ( 1 ) radial_component ( 0 ) rotational_component ( 0 ) orthogonal_component ( 0 ) scale_components ( on ) velocity_noise ( off ) velocity_freq ( 0.5 0.5 0.5 ) velocity_amp ( 1 ) magnitude_falloff ( off ) velocity_falloff ( 2 ) velocity_falloff1pos ( 0.63636362552642822 ) velocity_falloff1value ( 0.95454543828964233 ) velocity_falloff1interp ( catmull-rom ) velocity_falloff2pos ( 1 ) velocity_falloff2value ( 0 ) velocity_falloff2interp ( catmull-rom )
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

# Node density_output (Vop/bind)
opadd -e -n bind density_output
oplocate -x 26.125800000000002 -y 1.2933399999999999 density_output
opspareds "" density_output
opparm -V 14.0.254 density_output parmname ( output ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( whenconnected ) exportcontext ( cvex )
chlock density_output -*
chautoscope density_output -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 density_output
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on density_output
opwire -n twoway4 -0 density_output
opexprlanguage -s hscript density_output
opuserdata -n '___Version___' -v '14.0.254' density_output

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
opwire -n particle_index1 -3 get_ang_attribute
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
opwire -n particle_index1 -3 get_orient_attribute
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
opwire -n particle_index1 -3 get_rot_attribute
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

# Node particle_index1 (Vop/bind)
opadd -e -n bind particle_index1
oplocate -x -22.675699999999999 -y 13.854100000000001 particle_index1
opspareds "" particle_index1
opparm -V 14.0.254 particle_index1 parmname ( pindex ) parmtype ( int ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_index1 -*
chautoscope particle_index1 -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_index1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_index1
opexprlanguage -s hscript particle_index1
opuserdata -n '___Version___' -v '14.0.254' particle_index1
opcf ..

# Node divide1 (Vop/divide)
opadd -e -n divide divide1
oplocate -x -21.139099999999999 -y 8.2904300000000006 divide1
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
oplocate -x -17.750399999999999 -y 7.5322500000000003 direction
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
opwire -n add3 -2 3d_noise
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
opparm -V 14.0.254 Excavation parmscope ( shaderparm ) parmaccess ( "" ) parmname ( excavation ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Excavation ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.070000000000000007 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
opparm -V 14.0.254 Falloff parmscope ( shaderparm ) parmaccess ( "" ) parmname ( falloff ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Falloff ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.10000000000000001 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
opparm -V 14.0.254 Noise_Treatment parmscope ( shaderparm ) parmaccess ( "" ) parmname ( treatment ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Noise Treatment' ) showlabel ( on ) parmtype ( int ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( on ) menuchoices ( '0 "Abs" 1 "1 - Abs" 2 "Clamp"' ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
oplocate -x -18.146100000000001 -y 10.654199999999999 Noise_Lookup
opspareds "" Noise_Lookup
opparm -V 14.0.254 Noise_Lookup parmscope ( shaderparm ) parmaccess ( "" ) parmname ( lookup ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Noise Lookup' ) showlabel ( on ) parmtype ( int ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( on ) menuchoices ( '0 "Displacement" 1 "World Space" 2 "Local Space" 3 "Unit Space"' ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
opparm -V 14.0.254 Frequency parmscope ( shaderparm ) parmaccess ( "" ) parmname ( freq ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Frequency ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 1.74 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 10 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
oplocate -x -13.045400000000001 -y 0.49557499999999999 Roughness
opspareds "" Roughness
opparm -V 14.0.254 Roughness parmscope ( shaderparm ) parmaccess ( "" ) parmname ( rough ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Roughness ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.5 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Roughness -*
chautoscope Roughness -*
opcolor -c 1 1 0.40000000596046448 Roughness
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Roughness
opexprlanguage -s hscript Roughness
opuserdata -n '___Version___' -v '14.0.254' Roughness

# Node Ocataves (Vop/parameter)
opadd -e -n parameter Ocataves
oplocate -x -11.501799999999999 -y -0.22595299999999999 Ocataves
opspareds "" Ocataves
opparm -V 14.0.254 Ocataves parmscope ( shaderparm ) parmaccess ( "" ) parmname ( ocataves ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Ocataves ) showlabel ( on ) parmtype ( int ) parmtypename ( "" ) floatdef ( 1.74 ) intdef ( 2 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Ocataves -*
chautoscope Ocataves -*
opcolor -c 1 1 0.40000000596046448 Ocataves
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Ocataves
opexprlanguage -s hscript Ocataves
opuserdata -n '___Version___' -v '14.0.254' Ocataves

# Node Offset (Vop/parameter)
opadd -e -n parameter Offset
oplocate -x -20.330300000000001 -y 7.0838299999999998 Offset
opspareds "" Offset
opparm -V 14.0.254 Offset parmscope ( shaderparm ) parmaccess ( "" ) parmname ( offset ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Offset ) showlabel ( on ) parmtype ( vector ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
opparm -V 14.0.254 Billowing_Speed parmscope ( shaderparm ) parmaccess ( "" ) parmname ( billowingspeed ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Billowing Speed' ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.10000000000000001 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ billowing != 1 } { process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
oplocate -x -13.0192 -y 5.7475399999999999 4d_offset
opspareds "" 4d_offset
opparm 4d_offset vec ( 0 0 0 ) fval4 ( 0 )
chlock 4d_offset -*
chautoscope 4d_offset -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 4d_offset
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on 4d_offset
opwire -n add3 -0 4d_offset
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
opparm -V 14.0.254 Billowing_Motion parmscope ( shaderparm ) parmaccess ( "" ) parmname ( billowing ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Billowing Motion' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
opparm -V 14.0.254 Swirling_Motion parmscope ( shaderparm ) parmaccess ( "" ) parmname ( swirling ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Swirling Motion' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
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
opparm -V 14.0.254 Swirling_Speed parmscope ( shaderparm ) parmaccess ( "" ) parmname ( swirlingspeed ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Swirling Speed' ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0.10000000000000001 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ swirling != 1 } { process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Swirling_Speed -*
chautoscope Swirling_Speed -*
opcolor -c 1 1 0.40000000596046448 Swirling_Speed
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Swirling_Speed
opexprlanguage -s hscript Swirling_Speed
opuserdata -n '___Version___' -v '14.0.254' Swirling_Speed

# Node multiply2 (Vop/multiply)
opadd -e -n multiply multiply2
oplocate -x -18.470199999999998 -y 4.5429599999999999 multiply2
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
oplocate -x -21.989000000000001 -y 11.119 add2
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
oplocate -x -14.2043 -y 1.64795 Amplitude
opspareds "" Amplitude
opparm -V 14.0.254 Amplitude parmscope ( shaderparm ) parmaccess ( "" ) parmname ( amp ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Amplitude ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 1 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 2 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_density != 1 } { modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock Amplitude -*
chautoscope Amplitude -*
opcolor -c 1 1 0.40000000596046448 Amplitude
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on Amplitude
opexprlanguage -s hscript Amplitude
opuserdata -n '___Version___' -v '14.0.254' Amplitude

# Node get_v (Vop/getattrib)
opadd -e -n getattrib get_v
oplocate -x -18.477699999999999 -y -7.1693800000000003 get_v
opspareds "" get_v
opparm get_v signature ( default ) opinput ( opinput:0 ) file ( '$HH/geo/defgeo.bgeo' ) atype ( point ) attrib ( v ) i1 ( 0 ) i2 ( 0 )
chlock get_v -*
chautoscope get_v -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 get_v
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on get_v
opwire -n particle_index1 -3 get_v
opexprlanguage -s hscript get_v
opuserdata -n '___Version___' -v '' get_v

# Node velocity_direction (Vop/normalize)
opadd -e -n normalize velocity_direction
oplocate -x -15.551600000000001 -y -9.4965100000000007 velocity_direction
opspareds "" velocity_direction
opparm velocity_direction signature ( default ) vec ( 1 0 0 ) vec_p ( 1 0 0 ) vec_v ( 1 0 0 ) vec_v4 ( 0 0 0 1 ) vec_un ( 1 0 0 ) vec_up ( 1 0 0 ) vec_uv ( 1 0 0 )
chlock velocity_direction -*
chautoscope velocity_direction -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 velocity_direction
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on velocity_direction
opwire -n -o 1 get_v -0 velocity_direction
opexprlanguage -s hscript velocity_direction
opuserdata -n '___Version___' -v '' velocity_direction

# Node voxel_pos (Vop/bind)
opadd -e -n bind voxel_pos
oplocate -x -22.834399999999999 -y -9.53749 voxel_pos
opspareds "" voxel_pos
opparm -V 14.0.254 voxel_pos parmname ( voxelpos ) parmtype ( float3 ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock voxel_pos -*
chautoscope voxel_pos -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 voxel_pos
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on voxel_pos
opexprlanguage -s hscript voxel_pos
opuserdata -n '___Version___' -v '14.0.254' voxel_pos

# Node particle_center (Vop/bind)
opadd -e -n bind particle_center
oplocate -x -23.430199999999999 -y -11.233700000000001 particle_center
opspareds "" particle_center
opparm -V 14.0.254 particle_center parmname ( pcenter ) parmtype ( vector ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_center -*
chautoscope particle_center -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_center
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_center
opexprlanguage -s hscript particle_center
opuserdata -n '___Version___' -v '14.0.254' particle_center

# Node local_pos (Vop/subtract)
opadd -e -n subtract local_pos
oplocate -x -20.295300000000001 -y -10.3446 local_pos
opspareds "" local_pos
opparm -V 14.0.254 local_pos
chlock local_pos -*
chautoscope local_pos -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 local_pos
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on local_pos
opwire -n voxel_pos -0 local_pos
opwire -n particle_center -1 local_pos
opexprlanguage -s hscript local_pos
opuserdata -n '___Version___' -v '14.0.254' local_pos

# Node particle_index1 (Vop/bind)
opadd -e -n bind particle_index1
oplocate -x -20.946400000000001 -y -6.7457799999999999 particle_index1
opspareds "" particle_index1
opparm -V 14.0.254 particle_index1 parmname ( pindex ) parmtype ( int ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_index1 -*
chautoscope particle_index1 -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_index1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_index1
opexprlanguage -s hscript particle_index1
opuserdata -n '___Version___' -v '14.0.254' particle_index1

# Node cross2 (Vop/cross)
opadd -e -n cross cross2
oplocate -x -9.2041400000000007 -y -9.7846499999999992 cross2
opspareds "" cross2
opparm cross2 signature ( default ) vec1 ( 1 0 0 ) vec2 ( 0 1 0 )
chlock cross2 -*
chautoscope cross2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 cross2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on cross2
opwire -n velocity_direction -0 cross2
opwire -n normalize4 -1 cross2
opexprlanguage -s hscript cross2
opuserdata -n '___Version___' -v '' cross2

# Node normalize4 (Vop/normalize)
opadd -e -n normalize normalize4
oplocate -x -16.188800000000001 -y -11.6378 normalize4
opspareds "" normalize4
opparm normalize4 signature ( default ) vec ( 1 0 0 ) vec_p ( 1 0 0 ) vec_v ( 1 0 0 ) vec_v4 ( 0 0 0 1 ) vec_un ( 1 0 0 ) vec_up ( 1 0 0 ) vec_uv ( 1 0 0 )
chlock normalize4 -*
chautoscope normalize4 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 normalize4
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on normalize4
opwire -n local_pos -0 normalize4
opexprlanguage -s hscript normalize4
opuserdata -n '___Version___' -v '' normalize4

# Node normalize5 (Vop/normalize)
opadd -e -n normalize normalize5
oplocate -x -6.6486299999999998 -y -9.64344 normalize5
opspareds "" normalize5
opparm normalize5 signature ( default ) vec ( 1 0 0 ) vec_p ( 1 0 0 ) vec_v ( 1 0 0 ) vec_v4 ( 0 0 0 1 ) vec_un ( 1 0 0 ) vec_up ( 1 0 0 ) vec_uv ( 1 0 0 )
chlock normalize5 -*
chautoscope normalize5 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 normalize5
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on normalize5
opwire -n cross2 -0 normalize5
opexprlanguage -s hscript normalize5
opuserdata -n '___Version___' -v '' normalize5

# Node dot1 (Vop/dot)
opadd -e -n dot dot1
oplocate -x -11.9146 -y -13.1912 dot1
opspareds "" dot1
opparm dot1 signature ( default ) vec1 ( 1 0 0 ) vec2 ( 0 1 0 ) vec1_c ( 1 0 0 ) vec2_c ( 0 1 0 ) vec1_uv ( 1 0 0 ) vec2_uv ( 0 1 0 ) vec1_uc ( 1 0 0 ) vec2_uc ( 0 1 0 )
chlock dot1 -*
chautoscope dot1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 dot1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on dot1
opwire -n velocity_direction -0 dot1
opwire -n normalize4 -1 dot1
opexprlanguage -s hscript dot1
opuserdata -n '___Version___' -v '' dot1

# Node multiply9 (Vop/multiply)
opadd -e -n multiply multiply9
oplocate -x -9.5316299999999998 -y -12.164999999999999 multiply9
opspareds "" multiply9
opparm -V 14.0.254 multiply9
chlock multiply9 -*
chautoscope multiply9 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply9
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply9
opwire -n velocity_direction -0 multiply9
opwire -n dot1 -1 multiply9
opexprlanguage -s hscript multiply9
opuserdata -n '___Version___' -v '14.0.254' multiply9

# Node subtract3 (Vop/subtract)
opadd -e -n subtract subtract3
oplocate -x -7.4253499999999999 -y -11.278499999999999 subtract3
opspareds "" subtract3
opparm -V 14.0.254 subtract3
chlock subtract3 -*
chautoscope subtract3 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 subtract3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on subtract3
opwire -n normalize4 -0 subtract3
opwire -n multiply9 -1 subtract3
opexprlanguage -s hscript subtract3
opuserdata -n '___Version___' -v '14.0.254' subtract3

# Node normalize6 (Vop/normalize)
opadd -e -n normalize normalize6
oplocate -x -5.5822500000000002 -y -11.670299999999999 normalize6
opspareds "" normalize6
opparm normalize6 signature ( default ) vec ( 1 0 0 ) vec_p ( 1 0 0 ) vec_v ( 1 0 0 ) vec_v4 ( 0 0 0 1 ) vec_un ( 1 0 0 ) vec_up ( 1 0 0 ) vec_uv ( 1 0 0 )
chlock normalize6 -*
chautoscope normalize6 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 normalize6
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on normalize6
opwire -n subtract3 -0 normalize6
opexprlanguage -s hscript normalize6
opuserdata -n '___Version___' -v '' normalize6

# Node new_velocity (Vop/add)
opadd -e -n add new_velocity
oplocate -x 8.8963800000000006 -y -9.1233299999999993 new_velocity
opspareds "" new_velocity
opparm -V 14.0.254 new_velocity
chlock new_velocity -*
chautoscope new_velocity -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 new_velocity
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on new_velocity
opwire -n multiply_velocity -0 new_velocity
opwire -n twoway1 -1 new_velocity
opexprlanguage -s hscript new_velocity
opuserdata -n '___Version___' -v '14.0.254' new_velocity

# Node multiply_velocity (Vop/multiply)
opadd -e -n multiply multiply_velocity
oplocate -x 5.4260700000000002 -y -6.8869800000000003 multiply_velocity
opspareds "" multiply_velocity
opparm -V 14.0.254 multiply_velocity
chlock multiply_velocity -*
chautoscope multiply_velocity -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply_velocity
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply_velocity
opwire -n -o 1 get_v -0 multiply_velocity
opwire -n velocity_scale -1 multiply_velocity
opexprlanguage -s hscript multiply_velocity
opuserdata -n '___Version___' -v '14.0.254' multiply_velocity

# Node velocity_scale (Vop/parameter)
opadd -e -n parameter velocity_scale
oplocate -x 2.9033199999999999 -y -7.4203799999999998 velocity_scale
opspareds "" velocity_scale
opparm -V 14.0.254 velocity_scale parmscope ( shaderparm ) parmaccess ( "" ) parmname ( velocity_scale ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Velocity Scale' ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 1 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( -2 2 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Scale the input velocity' )
chlock velocity_scale -*
chautoscope velocity_scale -*
opcolor -c 1 1 0.40000000596046448 velocity_scale
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on velocity_scale
opexprlanguage -s hscript velocity_scale
opuserdata -n '___Version___' -v '14.0.254' velocity_scale

# Node rotational_component (Vop/parameter)
opadd -e -n parameter rotational_component
oplocate -x -4.9459099999999996 -y -10.5966 rotational_component
opspareds "" rotational_component
opparm -V 14.0.254 rotational_component parmscope ( shaderparm ) parmaccess ( "" ) parmname ( rotational_component ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Rotational ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( -1 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Add velocity components that circulate about the input velocity direction.' )
chlock rotational_component -*
chautoscope rotational_component -*
opcolor -c 1 1 0.40000000596046448 rotational_component
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on rotational_component
opexprlanguage -s hscript rotational_component
opuserdata -n '___Version___' -v '14.0.254' rotational_component

# Node orthogonal_component (Vop/parameter)
opadd -e -n parameter orthogonal_component
oplocate -x -4.9473700000000003 -y -13.135300000000001 orthogonal_component
opspareds "" orthogonal_component
opparm -V 14.0.254 orthogonal_component parmscope ( shaderparm ) parmaccess ( "" ) parmname ( orthogonal_component ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Orthogonal ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( -1 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Add velocity components that are orthogonal to the input velocity direction.' )
chlock orthogonal_component -*
chautoscope orthogonal_component -*
opcolor -c 1 1 0.40000000596046448 orthogonal_component
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on orthogonal_component
opexprlanguage -s hscript orthogonal_component
opuserdata -n '___Version___' -v '14.0.254' orthogonal_component

# Node multiply6 (Vop/multiply)
opadd -e -n multiply multiply6
oplocate -x -2.68804 -y -9.9258400000000009 multiply6
opspareds "" multiply6
opparm -V 14.0.254 multiply6
chlock multiply6 -*
chautoscope multiply6 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply6
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply6
opwire -n normalize5 -0 multiply6
opwire -n rotational_component -1 multiply6
opexprlanguage -s hscript multiply6
opuserdata -n '___Version___' -v '14.0.254' multiply6

# Node multiply10 (Vop/multiply)
opadd -e -n multiply multiply10
oplocate -x -2.6898 -y -11.9527 multiply10
opspareds "" multiply10
opparm -V 14.0.254 multiply10
chlock multiply10 -*
chautoscope multiply10 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply10
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply10
opwire -n normalize6 -0 multiply10
opwire -n orthogonal_component -1 multiply10
opexprlanguage -s hscript multiply10
opuserdata -n '___Version___' -v '14.0.254' multiply10

# Node v_falloff (Vop/rampparm)
opadd -e -n rampparm v_falloff
oplocate -x -1.8596999999999999 -y -18.358499999999999 v_falloff
opspareds "" v_falloff
opparm -V 14.0.254 v_falloff parmscope ( shaderparm ) parmaccess ( "" ) parmname ( velocity_falloff ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Falloff ) ramptype ( flt ) rampcolortype ( rgb ) rampbasisdefault ( catmull-rom ) separator1 ( ) useasparmdefiner ( off ) separator2 ( ) rampshowcontrolsdefault ( off ) disablewhen ( '{ process_velocity != 1 } { magnitude_falloff != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock v_falloff -*
chautoscope v_falloff -*
opcolor -c 1 1 0.40000000596046448 v_falloff
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on v_falloff
opwire -n fit2 -0 v_falloff
opexprlanguage -s hscript v_falloff
opuserdata -n '___Version___' -v '14.0.254' v_falloff

# Node length2 (Vop/length)
opadd -e -n length length2
oplocate -x -9.5434900000000003 -y -16.484500000000001 length2
opspareds "" length2
opparm length2 signature ( default ) vec ( 1 1 1 ) vec_p ( 1 1 1 ) vec_n ( 1 1 1 ) vec_v4 ( 1 1 1 1 ) vec_uv ( 1 1 1 ) vec_up ( 1 1 1 ) vec_un ( 1 1 1 )
chlock length2 -*
chautoscope length2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 length2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on length2
opwire -n local_pos -0 length2
opexprlanguage -s hscript length2
opuserdata -n '___Version___' -v '' length2

# Node v_noise (Vop/aanoise)
opadd -e -n aanoise v_noise
oplocate -x -10.529400000000001 -y -20.1218 v_noise
opspareds "" v_noise
chblockbegin
chadd -t 0 0 v_noise freq1
chkey -t 0 -v 1 -m 0 -a 0 -A 0 -T a  -F 'ch(\'../v_freq/float3def1\')' v_noise/freq1
chadd -t 0 0 v_noise freq2
chkey -t 0 -v 1 -m 0 -a 0 -A 0 -T a  -F 'ch(\'../v_freq/float3def2\')' v_noise/freq2
chadd -t 0 0 v_noise freq3
chkey -t 0 -v 1 -m 0 -a 0 -A 0 -T a  -F 'ch(\'../v_freq/float3def3\')' v_noise/freq3
chadd -t 0 0 v_noise offset1
chkey -t 0 -v 0 -m 0 -a 0 -A 0 -T a  -F 'ch(\'../offset2/pointdef1\')' v_noise/offset1
chadd -t 0 0 v_noise offset2
chkey -t 0 -v 0 -m 0 -a 0 -A 0 -T a  -F 'ch(\'../offset2/pointdef2\')' v_noise/offset2
chadd -t 0 0 v_noise offset3
chkey -t 0 -v 0 -m 0 -a 0 -A 0 -T a  -F 'ch(\'../offset2/pointdef3\')' v_noise/offset3
chadd -t 0 0 v_noise amp
chkey -t 0 -v 1 -m 0 -a 0 -A 0 -T a  -F 'ch(\'../v_amp/floatdef\')' v_noise/amp
chblockend
opparm v_noise signature ( default ) pos_ff ( 0 ) pos ( 0 0 0 ) pos_fp ( 0 0 0 0 ) freq_ff ( 1 ) freq ( freq1 freq2 freq3 ) freq_fp ( 1 1 1 1 ) offset_ff ( 0 ) offset ( offset1 offset2 offset3 ) offset_fp ( 0 0 0 0 ) amp ( amp ) rough ( 0.5 ) maxoctave ( 8 ) noisetype ( noise )
chlock v_noise -*
chautoscope v_noise -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 v_noise
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on v_noise
opwire -n local_pos -0 v_noise
opwire -n v_freq -1 v_noise
opwire -n inttovec1 -2 v_noise
opwire -n v_amp -3 v_noise
opexprlanguage -s hscript v_noise
opuserdata -n '___Version___' -v '' v_noise

# Node particle_radius (Vop/bind)
opadd -e -n bind particle_radius
oplocate -x -9.6593099999999996 -y -17.616499999999998 particle_radius
opspareds "" particle_radius
opparm -V 14.0.254 particle_radius parmname ( pradius ) parmtype ( float ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_radius -*
chautoscope particle_radius -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_radius
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_radius
opexprlanguage -s hscript particle_radius
opuserdata -n '___Version___' -v '14.0.254' particle_radius

# Node fit2 (Vop/fit)
opadd -e -n fit fit2
oplocate -x -7.3133600000000003 -y -17.049299999999999 fit2
opspareds "" fit2
opparm fit2 signature ( default ) srcmin ( 0 ) srcmax ( 1 ) destmin ( 0 ) destmax ( 1 ) srcmin_v ( 0 0 0 ) srcmax_v ( 1 1 1 ) destmin_v ( 0 0 0 ) destmax_v ( 1 1 1 ) srcmin_p ( 0 0 0 ) srcmax_p ( 1 1 1 ) destmin_p ( 0 0 0 ) destmax_p ( 1 1 1 ) srcmin_n ( 0 0 0 ) srcmax_n ( 1 1 1 ) destmin_n ( 0 0 0 ) destmax_n ( 1 1 1 ) srcmin_c ( 0 0 0 ) srcmax_c ( 1 1 1 ) destmin_c ( 0 0 0 ) destmax_c ( 1 1 1 ) srcmin_v4 ( 0 0 0 0 ) srcmax_v4 ( 1 1 1 1 ) destmin_v4 ( 0 0 0 0 ) destmax_v4 ( 1 1 1 1 ) srcmin_uf ( 0 ) srcmax_uf ( 1 ) destmin_uf ( 0 ) destmax_uf ( 1 ) srcmin_uv ( 0 0 0 ) srcmax_uv ( 1 1 1 ) destmin_uv ( 0 0 0 ) destmax_uv ( 1 1 1 ) srcmin_up ( 0 0 0 ) srcmax_up ( 1 1 1 ) destmin_up ( 0 0 0 ) destmax_up ( 1 1 1 ) srcmin_un ( 0 0 0 ) srcmax_un ( 1 1 1 ) destmin_un ( 0 0 0 ) destmax_un ( 1 1 1 ) srcmin_uc ( 0 0 0 ) srcmax_uc ( 1 1 1 ) destmin_uc ( 0 0 0 ) destmax_uc ( 1 1 1 )
chlock fit2 -*
chautoscope fit2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 fit2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on fit2
opwire -n length2 -0 fit2
opwire -n particle_radius -2 fit2
opexprlanguage -s hscript fit2
opuserdata -n '___Version___' -v '' fit2

# Node fit3 (Vop/fit)
opadd -e -n fit fit3
oplocate -x 0.412609 -y -21.6983 fit3
opspareds "" fit3
opparm fit3 signature ( default ) srcmin ( -0.5 ) srcmax ( 0.5 ) destmin ( 0 ) destmax ( 1.5 ) srcmin_v ( 0 0 0 ) srcmax_v ( 1 1 1 ) destmin_v ( 0 0 0 ) destmax_v ( 1 1 1 ) srcmin_p ( 0 0 0 ) srcmax_p ( 1 1 1 ) destmin_p ( 0 0 0 ) destmax_p ( 1 1 1 ) srcmin_n ( 0 0 0 ) srcmax_n ( 1 1 1 ) destmin_n ( 0 0 0 ) destmax_n ( 1 1 1 ) srcmin_c ( 0 0 0 ) srcmax_c ( 1 1 1 ) destmin_c ( 0 0 0 ) destmax_c ( 1 1 1 ) srcmin_v4 ( 0 0 0 0 ) srcmax_v4 ( 1 1 1 1 ) destmin_v4 ( 0 0 0 0 ) destmax_v4 ( 1 1 1 1 ) srcmin_uf ( 0 ) srcmax_uf ( 1 ) destmin_uf ( 0 ) destmax_uf ( 1 ) srcmin_uv ( 0 0 0 ) srcmax_uv ( 1 1 1 ) destmin_uv ( 0 0 0 ) destmax_uv ( 1 1 1 ) srcmin_up ( 0 0 0 ) srcmax_up ( 1 1 1 ) destmin_up ( 0 0 0 ) destmax_up ( 1 1 1 ) srcmin_un ( 0 0 0 ) srcmax_un ( 1 1 1 ) destmin_un ( 0 0 0 ) destmax_un ( 1 1 1 ) srcmin_uc ( 0 0 0 ) srcmax_uc ( 1 1 1 ) destmin_uc ( 0 0 0 ) destmax_uc ( 1 1 1 )
chlock fit3 -*
chautoscope fit3 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 fit3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on fit3
opwire -n v_noise -0 fit3
opexprlanguage -s hscript fit3
opuserdata -n '___Version___' -v '' fit3

# Node multiply4 (Vop/multiply)
opadd -e -n multiply multiply4
oplocate -x 14.512 -y -9.6923899999999996 multiply4
opspareds "" multiply4
opparm -V 14.0.254 multiply4
chlock multiply4 -*
chautoscope multiply4 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply4
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply4
opwire -n new_velocity -0 multiply4
opwire -n twoway2 -1 multiply4
opwire -n twoway3 -2 multiply4
opexprlanguage -s hscript multiply4
opuserdata -n '___Version___' -v '14.0.254' multiply4

# Node twoway2 (Vop/twoway)
opadd -e -n twoway twoway2
oplocate -x 1.7657400000000001 -y -17.934899999999999 twoway2
opspareds "" twoway2
opparm twoway2 signature ( default ) condtype ( 0 ) input2 ( 1 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock twoway2 -*
chautoscope twoway2 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 twoway2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on twoway2
opwire -n magnitude_falloff -0 twoway2
opwire -n v_falloff -1 twoway2
opexprlanguage -s hscript twoway2
opuserdata -n '___Version___' -v '' twoway2

# Node magnitude_falloff (Vop/parameter)
opadd -e -n parameter magnitude_falloff
oplocate -x -0.52105699999999999 -y -16.3323 magnitude_falloff
opspareds "" magnitude_falloff
opparm -V 14.0.254 magnitude_falloff parmscope ( shaderparm ) parmaccess ( "" ) parmname ( magnitude_falloff ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Magnitude Falloff' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 1 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Apply radial velocity magnitude falloff' )
chlock magnitude_falloff -*
chautoscope magnitude_falloff -*
opcolor -c 1 1 0.40000000596046448 magnitude_falloff
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on magnitude_falloff
opexprlanguage -s hscript magnitude_falloff
opuserdata -n '___Version___' -v '14.0.254' magnitude_falloff

# Node twoway3 (Vop/twoway)
opadd -e -n twoway twoway3
oplocate -x 2.60581 -y -20.634 twoway3
opspareds "" twoway3
opparm twoway3 signature ( default ) condtype ( 0 ) input2 ( 1 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock twoway3 -*
chautoscope twoway3 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 twoway3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on twoway3
opwire -n velocity_noise -0 twoway3
opwire -n fit3 -1 twoway3
opexprlanguage -s hscript twoway3
opuserdata -n '___Version___' -v '' twoway3

# Node velocity_noise (Vop/parameter)
opadd -e -n parameter velocity_noise
oplocate -x 0.49780799999999997 -y -20.012499999999999 velocity_noise
opspareds "" velocity_noise
opparm -V 14.0.254 velocity_noise parmscope ( shaderparm ) parmaccess ( "" ) parmname ( velocity_noise ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Multiplicative Noise' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 1 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Scale final velocity using noise' )
chlock velocity_noise -*
chautoscope velocity_noise -*
opcolor -c 1 1 0.40000000596046448 velocity_noise
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on velocity_noise
opexprlanguage -s hscript velocity_noise
opuserdata -n '___Version___' -v '14.0.254' velocity_noise

# Node v_freq (Vop/parameter)
opadd -e -n parameter v_freq
oplocate -x -14.1884 -y -19.073899999999998 v_freq
opspareds "" v_freq
opparm -V 14.0.254 v_freq parmscope ( shaderparm ) parmaccess ( "" ) parmname ( velocity_freq ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Frequency ) showlabel ( on ) parmtype ( float3 ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0.5 0.5 0.5 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { velocity_noise != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Noise frequency' )
chlock v_freq -*
chautoscope v_freq -*
opcolor -c 1 1 0.40000000596046448 v_freq
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on v_freq
opexprlanguage -s hscript v_freq
opuserdata -n '___Version___' -v '14.0.254' v_freq

# Node v_amp (Vop/parameter)
opadd -e -n parameter v_amp
oplocate -x -14.1106 -y -20.877700000000001 v_amp
opspareds "" v_amp
opparm -V 14.0.254 v_amp parmscope ( shaderparm ) parmaccess ( "" ) parmname ( velocity_amp ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Amplitude ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 1 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( -2 2 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { velocity_noise != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Noise amplitude' )
chlock v_amp -*
chautoscope v_amp -*
opcolor -c 1 1 0.40000000596046448 v_amp
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on v_amp
opexprlanguage -s hscript v_amp
opuserdata -n '___Version___' -v '14.0.254' v_amp

# Node particle_index2 (Vop/bind)
opadd -e -n bind particle_index2
oplocate -x -17.847100000000001 -y -19.855899999999998 particle_index2
opspareds "" particle_index2
opparm -V 14.0.254 particle_index2 parmname ( pindex ) parmtype ( int ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_index2 -*
chautoscope particle_index2 -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_index2
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_index2
opexprlanguage -s hscript particle_index2
opuserdata -n '___Version___' -v '14.0.254' particle_index2

# Node inttovec1 (Vop/inttovec)
opadd -e -n inttovec inttovec1
oplocate -x -15.618600000000001 -y -19.9971 inttovec1
opspareds "" inttovec1
opparm inttovec1 int1 ( 0 ) int2 ( 0 ) int3 ( 0 )
chlock inttovec1 -*
chautoscope inttovec1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 inttovec1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on inttovec1
opwire -n particle_index2 -1 inttovec1
opexprlanguage -s hscript inttovec1
opuserdata -n '___Version___' -v '' inttovec1

# Node velocity_output (Vop/bind)
opadd -e -n bind velocity_output
oplocate -x 26.219999999999999 -y -2.5619900000000002 velocity_output
opspareds "" velocity_output
opparm -V 14.0.254 velocity_output parmname ( v ) parmtype ( float3 ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( on ) parmuniform ( on ) usebound ( off ) exportparm ( whenconnected ) exportcontext ( cvex )
chlock velocity_output -*
chautoscope velocity_output -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 velocity_output
opset -d on -r on -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on velocity_output
opwire -n twoway5 -0 velocity_output
opexprlanguage -s hscript velocity_output
opuserdata -n '___Version___' -v '14.0.254' velocity_output

# Node twoway4 (Vop/twoway)
opadd -e -n twoway twoway4
oplocate -x 18.808199999999999 -y 6.9044400000000001 twoway4
opspareds "" twoway4
opparm twoway4 signature ( default ) condtype ( 0 ) input2 ( 1 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock twoway4 -*
chautoscope twoway4 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 twoway4
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on twoway4
opwire -n process_density -0 twoway4
opwire -n clamp1 -1 twoway4
opexprlanguage -s hscript twoway4
opuserdata -n '___Version___' -v '' twoway4

# Node process_density (Vop/parameter)
opadd -e -n parameter process_density
oplocate -x 16.707599999999999 -y 7.8435600000000001 process_density
opspareds "" process_density
opparm -V 14.0.254 process_density parmscope ( shaderparm ) parmaccess ( "" ) parmname ( process_density ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Process Density' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ modeling != 1 } { createdensity != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( "" )
chlock process_density -*
chautoscope process_density -*
opcolor -c 1 1 0.40000000596046448 process_density
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on process_density
opexprlanguage -s hscript process_density
opuserdata -n '___Version___' -v '14.0.254' process_density

# Node twoway5 (Vop/twoway)
opadd -e -n twoway twoway5
oplocate -x 17.350000000000001 -y -8.1232799999999994 twoway5
opspareds "" twoway5
opparm twoway5 signature ( v ) condtype ( 0 ) input2 ( 1 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock twoway5 -*
chautoscope twoway5 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 twoway5
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on twoway5
opwire -n process_velocity -0 twoway5
opwire -n multiply4 -1 twoway5
opwire -n -o 1 get_v -2 twoway5
opexprlanguage -s hscript twoway5
opuserdata -n '___Version___' -v '' twoway5

# Node process_velocity (Vop/parameter)
opadd -e -n parameter process_velocity
oplocate -x 15.2494 -y -7.1841699999999999 process_velocity
opspareds "" process_velocity
opparm -V 14.0.254 process_velocity parmscope ( shaderparm ) parmaccess ( "" ) parmname ( process_velocity ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Process Velocity' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( 0 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Requires a velocity point attribute named \'v\'.' )
chlock process_velocity -*
chautoscope process_velocity -*
opcolor -c 1 1 0.40000000596046448 process_velocity
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on process_velocity
opexprlanguage -s hscript process_velocity
opuserdata -n '___Version___' -v '14.0.254' process_velocity

# Node radial_component (Vop/parameter)
opadd -e -n parameter radial_component
oplocate -x -4.8677099999999998 -y -15.0085 radial_component
opspareds "" radial_component
opparm -V 14.0.254 radial_component parmscope ( shaderparm ) parmaccess ( "" ) parmname ( radial_component ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( Radial ) showlabel ( on ) parmtype ( float ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( -1 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Add velocity components that diverge or converge from the center.' )
chlock radial_component -*
chautoscope radial_component -*
opcolor -c 1 1 0.40000000596046448 radial_component
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on radial_component
opexprlanguage -s hscript radial_component
opuserdata -n '___Version___' -v '14.0.254' radial_component

# Node multiply11 (Vop/multiply)
opadd -e -n multiply multiply11
oplocate -x -2.5985200000000002 -y -14.122299999999999 multiply11
opspareds "" multiply11
opparm -V 14.0.254 multiply11
chlock multiply11 -*
chautoscope multiply11 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply11
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply11
opwire -n normalize4 -0 multiply11
opwire -n radial_component -1 multiply11
opexprlanguage -s hscript multiply11
opuserdata -n '___Version___' -v '14.0.254' multiply11

# Node new_velocity_components (Vop/add)
opadd -e -n add new_velocity_components
oplocate -x 0.20522299999999999 -y -12.4038 new_velocity_components
opspareds "" new_velocity_components
opparm -V 14.0.254 new_velocity_components
chlock new_velocity_components -*
chautoscope new_velocity_components -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 new_velocity_components
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on new_velocity_components
opwire -n multiply6 -0 new_velocity_components
opwire -n multiply10 -1 new_velocity_components
opwire -n multiply11 -2 new_velocity_components
opexprlanguage -s hscript new_velocity_components
opuserdata -n '___Version___' -v '14.0.254' new_velocity_components

# Node scale_components (Vop/parameter)
opadd -e -n parameter scale_components
oplocate -x 2.6542500000000002 -y -9.0706399999999991 scale_components
opspareds "" scale_components
opparm -V 14.0.254 scale_components parmscope ( shaderparm ) parmaccess ( "" ) parmname ( scale_components ) parmprefix ( "" ) parmpostfix ( "" ) parmlabel ( 'Scale Radial, Rotational and Orthogonal Components ' ) showlabel ( on ) parmtype ( toggle ) parmtypename ( "" ) floatdef ( 0 ) intdef ( 0 ) toggledef ( on ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) rangeflt ( -1 1 ) rangeint ( 0 10 ) stringtype ( off ) opfilter ( !!OBJ/LIGHT!! ) parmcomment ( "" ) separator1 ( ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex ) providemenu ( off ) menuchoices ( "" ) separator2 ( ) invisible ( off ) joinnext ( off ) disablewhen ( '{ process_velocity != 1 } { modeling != 1 }' ) hidewhen ( "" ) callback ( "" ) help ( 'Scale radial, rotational and orthogonal components with input velocity magnitude.' )
chlock scale_components -*
chautoscope scale_components -*
opcolor -c 1 1 0.40000000596046448 scale_components
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on scale_components
opexprlanguage -s hscript scale_components
opuserdata -n '___Version___' -v '14.0.254' scale_components

# Node length3 (Vop/length)
opadd -e -n length length3
oplocate -x -1.6192599999999999 -y -8.5905100000000001 length3
opspareds "" length3
opparm length3 signature ( default ) vec ( 1 1 1 ) vec_p ( 1 1 1 ) vec_n ( 1 1 1 ) vec_v4 ( 1 1 1 1 ) vec_uv ( 1 1 1 ) vec_up ( 1 1 1 ) vec_un ( 1 1 1 )
chlock length3 -*
chautoscope length3 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 length3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on length3
opwire -n -o 1 get_v -0 length3
opexprlanguage -s hscript length3
opuserdata -n '___Version___' -v '' length3

# Node twoway1 (Vop/twoway)
opadd -e -n twoway twoway1
oplocate -x 5.3993900000000004 -y -10.4575 twoway1
opspareds "" twoway1
opparm twoway1 signature ( v ) condtype ( 0 ) input2 ( 0 ) input2_i ( 0 ) input2_s ( "" ) input2_v ( 0 0 0 ) input2_p ( 0 0 0 ) input2_n ( 0 0 0 ) input2_c ( 1 1 1 ) input2_v4 ( 0 0 0 0 ) input2_m3 ( 1 0 0 0 1 0 0 0 1 ) input2_m ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) input2_uf ( 0 ) input2_uv ( 0 0 0 ) input2_up ( 0 0 0 ) input2_un ( 0 0 0 ) input2_uc ( 0 0 0 ) input2_um ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 )
chlock twoway1 -*
chautoscope twoway1 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 twoway1
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on twoway1
opwire -n scale_components -0 twoway1
opwire -n multiply7 -1 twoway1
opwire -n new_velocity_components -2 twoway1
opexprlanguage -s hscript twoway1
opuserdata -n '___Version___' -v '' twoway1

# Node multiply7 (Vop/multiply)
opadd -e -n multiply multiply7
oplocate -x 2.3395700000000001 -y -10.4575 multiply7
opspareds "" multiply7
opparm -V 14.0.254 multiply7
chlock multiply7 -*
chautoscope multiply7 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 multiply7
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on multiply7
opwire -n new_velocity_components -0 multiply7
opwire -n length3 -1 multiply7
opexprlanguage -s hscript multiply7
opuserdata -n '___Version___' -v '14.0.254' multiply7

# Node particle_index3 (Vop/bind)
opadd -e -n bind particle_index3
oplocate -x -21.4239 -y 6.1787400000000003 particle_index3
opspareds "" particle_index3
opparm -V 14.0.254 particle_index3 parmname ( pindex ) parmtype ( int ) floatdef ( 0 ) intdef ( 0 ) toggledef ( off ) angledef ( 0 ) logfloatdef ( 0 ) float2def ( 0 0 ) float3def ( 0 0 0 ) vectordef ( 0 0 0 ) normaldef ( 0 0 0 ) pointdef ( 0 0 0 ) directiondef ( 1 0 0 ) float4def ( 0 0 0 0 ) floatm2def ( 1 0 0 1 ) float9def ( 1 0 0 0 1 0 0 0 1 ) float16def ( 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ) stringdef ( "" ) filedef ( "" ) imagedef ( "" ) geometrydef ( "" ) colordef ( 0 0 0 ) color4def ( 0 0 0 0 ) coshaderdef ( "" ) floatadef ( ) intadef ( ) vector2adef ( ) vectoradef ( ) float4adef ( ) floatm2adef ( ) float9adef ( ) float16adef ( ) stringadef ( ) coshaderadef ( "" ) structdef ( "" ) useasparmdefiner ( off ) parmuniform ( on ) usebound ( off ) exportparm ( off ) exportcontext ( cvex )
chlock particle_index3 -*
chautoscope particle_index3 -*
opcolor -c 0.60000002384185791 0.60000002384185791 1 particle_index3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on particle_index3
opexprlanguage -s hscript particle_index3
opuserdata -n '___Version___' -v '14.0.254' particle_index3

# Node add3 (Vop/add)
opadd -e -n add add3
oplocate -x -17.600000000000001 -y 6.3162500000000001 add3
opspareds "" add3
opparm -V 14.0.254 add3
chlock add3 -*
chautoscope add3 -*
opcolor -c 0.80000001192092896 0.80000001192092896 0.80000001192092896 add3
opset -d off -r off -h off -f off -y off -t off -l off -s off -u off -c off -e on -b off -L off -M off -H on add3
opwire -n Offset -0 add3
opwire -n particle_index3 -1 add3
opexprlanguage -s hscript add3
opuserdata -n '___Version___' -v '14.0.254' add3
opcf Inputs
oporder -e voxel_pos voxel_size particle_center particle_radius particle_index Frame Time TimeInc suboutput1 subinput1 length1 split spherical_coord atan1 divide1 trig1 get_ang_attribute quaternion1 qrotate1 subtract1 P get_orient_attribute qrotate2 get_rot_attribute qrotate3 rotation1 rotation2 rotation3 CP particle_index1 
opcf ..

# Sticky Note modify_any_point_attribute

python -c 'hou.pwd().createStickyNote("modify_any_point_attribute")'

python -c 'hou.pwd().findStickyNote("modify_any_point_attribute").setColor(hou.Color([1, 0.969, 0.522]))'

python -c 'hou.pwd().findStickyNote("modify_any_point_attribute").setText("All output grids can be modified using VEX. ")'

python -c 'hou.pwd().findStickyNote("modify_any_point_attribute").setPosition(hou.Vector2(28.5342, -1.28699))'

python -c 'hou.pwd().findStickyNote("modify_any_point_attribute").setSize(hou.Vector2(2.71573, 2.2475))'

python -c 'hou.pwd().findStickyNote("modify_any_point_attribute").setMinimized(False)'
oporder -e density_output Inputs divide1 direction 3d_noise abs1 pow1 length1 subtract1 clamp1 Excavation fit1 Falloff scale_falloff negate_falloff min_clamp_falloff complement1 clamp2 noise_value Noise_Treatment 3d_pos Noise_Lookup Frequency 3d_frequency Roughness Ocataves Offset Billowing_Speed multiply1 switch1 vectofloat1 normalize_theta normalize_phi add1 frompolar1 4d_offset 4d_frequency 4d_pos Billowing_Motion Swirling_Motion Swirling_Speed multiply2 4d_noise noise_val add2 Amplitude get_v velocity_direction voxel_pos particle_center local_pos particle_index1 cross2 normalize4 normalize5 dot1 multiply9 subtract3 normalize6 new_velocity multiply_velocity velocity_scale rotational_component orthogonal_component multiply6 multiply10 v_falloff length2 v_noise particle_radius fit2 fit3 multiply4 twoway2 magnitude_falloff twoway3 velocity_noise v_freq v_amp particle_index2 inttovec1 velocity_output twoway4 process_density twoway5 process_velocity radial_component multiply11 new_velocity_components scale_components length3 twoway1 multiply7 particle_index3 add3 

opcf ..
