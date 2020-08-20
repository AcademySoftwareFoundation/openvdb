#!/bin/bash
EXE=$1
shift
ARGS=()
while [ $# -gt 0 ]; do
	a=$1
	case $a in
	/*)
		if [ -d "$(dirname $a)" ] && [ "$(dirname $a)" != "/" ]; then
			a=z:$a
		fi
		;;
	*)
		;;
	esac
	ARGS+=("$a")
	shift
done
wine "$EXE" "${ARGS[@]}" 2> >(grep -v '^[[:alnum:]]*:\?fixme' | grep -v ^err:bcrypt:hash_init | sed 's/\r//' >&2) | sed 's/\r//' | sed 's/z:\([\\/]\)/\1/i'
exit $PIPESTATUS
