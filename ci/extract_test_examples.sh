#!/usr/bin/env bash
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../"
VDB_AX=$ROOT/build/openvdb_ax/openvdb_ax/cmd/vdb_ax

DOCS=()
DOCS+=($ROOT/doc/ax/ax.txt)
DOCS+=($ROOT/doc/ax/axcplusplus.txt)
DOCS+=($ROOT/doc/ax/axexamples.txt)
# DOCS+=($ROOT/doc/ax/axfunctionlist.txt) do not test this file
DOCS+=($ROOT/doc/ax/doc.txt)

for DOC in "${DOCS[@]}"; do
    echo "Checking doxygen code in '$DOC...'"
    # Make sure the file exists
    if [ ! -f $DOC ]; then
        echo "Could not find '$DOC.'"
        exit -1
    fi

    # Extract all code segments from doxygen documentation in between @code and @endcode
    data=$(sed -n '/^ *@code.* *$/,/^ *@endcode *$/p' $DOC)

    str=""
    skip=false
    count=0
    # For each extracted line, rebuild each code segment from @code and @endcode and
    # run it through a vdb_ax binary if necessary
    while IFS= read -r line; do
        # If the code is marked as unparsed, c++ or shell, skip
        if [[ "$line" == "@code{.unparsed}" ]]; then
            skip=true
        elif [[ "$line" == "@code{.cpp}" ]]; then
            skip=true
        elif [[ "$line" == "@code{.sh}" ]]; then
            skip=true
        elif [[ $line == @code* ]]; then
            str=""
        elif [[ $line == @endcode ]]; then
            echo -e "\nTesting [$count]:"
            if [ "$skip" = true ]; then
                echo "Skipping the following unparsed/c++/shell code:"
                echo -e "$str" | sed 's/^/    /'
            else
                # parse
                $VDB_AX analyze -s "$str" -v
            fi
            skip=false
            count=$((count+1))
            str=""
        else
            str+="$line"$'\n'
        fi

    done <<< "$data"
done
