#!/bin/bash
#
# Batch retraining cron job setup script
# Sets up a weekly cron job to trigger batch retraining every Sunday at 2 AM ET
#
# Usage:
#   bash scripts/setup_cron.sh
#

set -e

PROJECT_ROOT="/Users/sayalimoghe/Documents/Career/GitHub/conversational-book-recommendation-agent"
PYTHON_EXECUTABLE=$(which python3)
CRON_SCHEDULE="0 7 * * 0"  # 7 AM UTC = 2 AM ET (accounting for EDT)
CRON_JOB_NAME="batch-retrain-cf-model"

echo "=========================================="
echo "Batch Retraining Cron Job Setup"
echo "=========================================="
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Python Executable: $PYTHON_EXECUTABLE"
echo "Schedule: $CRON_SCHEDULE (every Sunday at 2 AM ET)"
echo ""

# Create a wrapper script that will be executed by cron
WRAPPER_SCRIPT="$PROJECT_ROOT/scripts/retrain_cron_wrapper.sh"

cat > "$WRAPPER_SCRIPT" << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper script for cron execution of batch retraining
# This script handles environment setup and logging

PROJECT_ROOT="/Users/sayalimoghe/Documents/Career/GitHub/conversational-book-recommendation-agent"
PYTHON_EXECUTABLE=$(which python3)

cd "$PROJECT_ROOT"

# Run batch retraining with full paths and error handling
{
    echo "================================================"
    echo "Batch Retrain Cron Job Execution"
    echo "Started at: $(date)"
    echo "================================================"
    
    "$PYTHON_EXECUTABLE" -m scripts.batch_retrain
    
    EXIT_CODE=$?
    
    echo "================================================"
    echo "Completed at: $(date)"
    echo "Exit Code: $EXIT_CODE"
    echo "================================================"
} >> "$PROJECT_ROOT/logs/eval_logs/batch_retrain_cron.log" 2>&1
WRAPPER_EOF

chmod +x "$WRAPPER_SCRIPT"

echo "✓ Created wrapper script: $WRAPPER_SCRIPT"
echo ""

# Check if cron job already exists
CRON_ENTRY="0 7 * * 0 $WRAPPER_SCRIPT"

if crontab -l 2>/dev/null | grep -q "$CRON_JOB_NAME"; then
    echo "⚠ A cron job for '$CRON_JOB_NAME' already exists"
    echo ""
    echo "Current cron jobs:"
    crontab -l 2>/dev/null | grep -i retrain || echo "  (none found)"
    echo ""
    echo "To update/remove the existing job, run:"
    echo "  crontab -e"
else
    echo "Adding cron job..."
    
    # Add to crontab if it doesn't exist
    (crontab -l 2>/dev/null; echo "# Batch retraining for CF model - Sunday 2 AM ET"; echo "$CRON_ENTRY") | crontab -
    
    echo "✓ Cron job added successfully!"
    echo ""
    echo "Cron job details:"
    echo "  Schedule: 0 7 * * 0 (every Sunday at 7 AM UTC / 2 AM ET)"
    echo "  Script: $WRAPPER_SCRIPT"
    echo "  Logs: $PROJECT_ROOT/logs/eval_logs/batch_retrain_cron.log"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To verify the cron job:"
echo "  crontab -l | grep retrain"
echo ""
echo "To manually run batch retraining:"
echo "  python3 scripts/batch_retrain.py"
echo ""
echo "To remove the cron job:"
echo "  crontab -e  (then delete the line)"
echo ""
