function darknet_git_revision() {
    darknet version | sed -n 's/.*-g\([0-9a-f]\+\).*/\1/p'
}
