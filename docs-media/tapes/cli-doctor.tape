# project: image-quality
# user: ines
# alt: Draft: a terminal runs qa doctor in the camera ISP's folder.
#
Source "tapes/_settings.tape"
Output out/cli/doctor/tape.mp4
Output out/cli/doctor/tape.webm

Hide
Type "cd ~/kitefin/camera-isp && clear"
Enter
Show
Sleep 800ms
Type "qa doctor"
Sleep 700ms
Enter
Hide
Wait@90s /\$$/
Show
Sleep 6s
