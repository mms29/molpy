mol load pdb 5ftm.pdb
set protein [atomselect top protein]
set chains [lsort -unique [$protein get pfrag]]
foreach chain $chains {
	set sel [atomselect top "pfrag $chain"]
	$sel writepdb 5ftm${chain}.pdb
}
#package require psfgen
#topology /home/guest/toppar/top_all36_prot.rtf
#pdbalias residue HIS HSE
#pdbalias atom ILE CD1 CD
#segment U {pdb AK.pdb}
#coordpdb AK.pdb U
#writepdb ubq.pdb
#writepsf ubq.psf
#exit
