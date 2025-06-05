#!/bin/bash

correct=0
incorrect=0

for file in spam_2/*; do
    output=$(python3 main.py predict spam "$file")

    if [[ "$output" == *"positive"* ]]; then
        ((correct++))
    elif [[ "$output" == *"negative"* ]]; then
        ((incorrect++))
    fi
done

for file in easy_ham_2/*; do
    output=$(python3 main.py predict spam "$file")

    if [[ "$output" == *"positive"* ]]; then
        ((incorrect++))
    elif [[ "$output" == *"negative"* ]]; then
        ((correct++))
    fi
done

echo "Correct: $correct"
echo "Incorrect: $incorrect"
