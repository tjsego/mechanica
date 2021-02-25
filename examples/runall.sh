#!/bin/sh
python=$1

echo $python

$($python boundary_uptake_secrete.py)
$($python argon.py)
$($python colormaps.py)
$($python epiboly1.py)
$($python flux_secrete.py)
$($python uptake_secrete.py)
$($python nacl.py)
$($python species_update.py)
#$($python tsphere.py)
$($python argon_tstat.py)
$($python cube.py)
$($python epiboly1_plotting.py)
$($python fluxtest1.py)
$($python phiplot.py)
$($python sphere.py)
$($python two-type.py)
$($python become_1.py)
$($python cube2.py)
$($python epiboly2.py)
$($python fluxtest2.py)
$($python plot.py)
$($python sphere_test.py)
$($python two_particle_tstat.py)
#$($python bond_ball.py)
$($python data_output.py)
$($python epiboly2_plotting.py)
$($python gargon.py)
$($python poiseuille.py)
$($python sphericalplot.py)
$($python type_colors.py)
$($python bonded_beads.py)
$($python dpd1.py)
$($python epiboly3.py)
$($python glj-1.py)
$($python rcube.py)
$($python spin.py)
$($python type_colors2.py)
$($python bonded_sc_lattice.py)
$($python dpd2.py)
$($python epiboly3_plot.py)
$($python glj.py)
$($python ring.py)
$($python spin2.py)
$($python version.py)
$($python bonded_sphere.py)
$($python dpd2new.py)
$($python epiboly_cluster.py)
$($python glj_cluster1.py)
$($python ring_ball.py)
$($python spin3.py)
$($python virus_docking.py)
$($python boundary1.py)
#$($python dpd3.py)
$($python epiboly_plotting.py)
$($python hex_lattice.py)
$($python square_well.py)
#$($python widget_bcc.py)
$($python cell_sorting.py)
$($python epiboly-asym.py)
$($python events1.py)
$($python mitosis.dyn.py)
#$($python shell_fill1.py)
$($python test.py)
$($python windowless.py)
$($python change_type.py)
$($python epiboly1.periodic.py)
$($python events2.py)
$($python mitosis.od.py)
$($python species.py)
$($python threaded_windowless.py)
$($python writedata_events.py)
