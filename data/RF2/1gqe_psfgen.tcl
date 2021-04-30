mol load pdb data/RF2/1gqe_fitted.pdb
set protein [atomselect top protein]
set chains [lsort -unique [$protein get pfrag]]
foreach chain $chains {
    set sel [atomselect top "pfrag $chain"]
    $sel writepdb data/RF2/1gqe_tmp${chain}.pdb
}
package require psfgen
topology data/toppar/top_all36_prot.rtf
pdbalias residue HIS HSE
pdbalias atom ILE CD1 CD
foreach chain $chains {
    segment U${chain} {pdb data/RF2/1gqe_tmp${chain}.pdb}
    coordpdb data/RF2/1gqe_tmp${chain}.pdb U${chain}
    rm -f data/RF2/1gqe_tmp${chain}.pdb U${chain}
}
guesscoord
writepdb data/RF2/1gqe_PSF.pdb
writepsf data/RF2/1gqe.psf
exit
