#!/bin/bash

# Set the local and remote paths
LOCAL_PATH="/sync/"
REMOTE_PATH="remote:cc/"
LOCAL_CHECKPOINTS_PATH="/sync/checkpoints"
REMOTE_CHECKPOINTS_PATH="remote:cc/checkpoints"

# Set the sync interval (in seconds)
SYNC_INTERVAL=1200

# Function to find the latest file in the local checkpoints folder
find_latest_checkpoint() {
    latest_file=$(find "$LOCAL_CHECKPOINTS_PATH" -type f -printf '%T+ %p\n' | sort -r | head -n1 | cut -d' ' -f2-)
    echo "$latest_file"
}

# Infinite loop to run the sync periodically
while true; do
    # Sync the non-checkpoint files
    echo "Syncing non-checkpoint files from $LOCAL_PATH to $REMOTE_PATH..."
    rclone sync --ignore-checksum --exclude "/checkpoints/**" "$LOCAL_PATH" "$REMOTE_PATH"
    echo "Non-checkpoint files sync complete."

    # Find the latest checkpoint file
    latest_checkpoint=$(find_latest_checkpoint)

    if [ -n "$latest_checkpoint" ]; then
        echo "Syncing the latest checkpoint \"$latest_checkpoint\" to $REMOTE_CHECKPOINTS_PATH..."
        rclone copy --ignore-checksum "$latest_checkpoint" "$REMOTE_CHECKPOINTS_PATH"
        echo "Latest checkpoint sync complete."
    else
        echo "No checkpoint files found in $LOCAL_CHECKPOINTS_PATH"
    fi

    echo "Waiting $SYNC_INTERVAL seconds before syncing again..."
    sleep $SYNC_INTERVAL
done
