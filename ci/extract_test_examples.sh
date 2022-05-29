#!/usr/bin/env bash
#################################################################################
# This script extracts all code blocks from AX documentation which are NOT      #
# marked as cpp/sh/unparsed and attempts to parse or compile them through the   #
# vdb_ax command line binary. Code blocks can be marked with a preceding        #
# notation to determine compilation mode:                                       #
#    <!--- P --->  = points                                                     #
#    <!--- V --->  = volumes                                                    #
#    <!--- A --->  = all                                                        #
# If not marked, code is only parsed (though ideally all code blocks should be  #
# compiled)                                                                     #
#################################################################################
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../"
VDB_AX=$ROOT/build/openvdb_cmd/vdb_ax/vdb_ax

DOCS=()
DOCS+=($ROOT/doc/ax/ax.txt)
DOCS+=($ROOT/doc/ax/axcplusplus.txt)
DOCS+=($ROOT/doc/ax/axexamples.txt)
# DOCS+=($ROOT/doc/ax/axfunctionlist.txt) do not test this file
DOCS+=($ROOT/doc/ax/doc.txt)

uncompiled=0

for DOC in "${DOCS[@]}"; do
    echo "Checking doxygen code in '$DOC...'"
    # Make sure the file exists
    if [ ! -f $DOC ]; then
        echo "Could not find '$DOC.'"
        exit -1
    fi

    # Extract all code segments from doxygen documentation in between @code and @endcode
    data=$(sed -n '/@code.* *$/,/^ *@endcode *$/p' $DOC)

    str=""
    compile=""
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
            compile="none"
        elif [[ $line == "<!-- V -->@code"* ]]; then
            str=""
            compile="volumes"
        elif [[ $line == "<!-- P -->@code"* ]]; then
            str=""
            compile="points"
        elif [[ $line == "<!-- A -->@code"* ]]; then
            str=""
            compile=""
        elif [[ $line == @endcode ]]; then
            echo -e "\nTesting [$count]:"
            if [ "$skip" = true ]; then
                echo "Skipping the following unparsed/c++/shell code:"
                echo -e "$str" | sed 's/^/    /'
            else
                # parse
                if [ "$compile" = "none" ]; then
                    $VDB_AX analyze -s "$str" -v
                    uncompiled=$((uncompiled+1))
                else
                    $VDB_AX analyze -s "$str" --try-compile $compile -v
                fi
            fi
            skip=false
            count=$((count+1))
            str=""
            compile="none"
        else
            str+="$line"$'\n'
        fi
    done <<< "$data"
done

echo ""
echo "Extract test examples completed successfully"
if [ $uncompiled -gt 0 ]; then
    echo " with $uncompiled uncompiled tests"
fi
