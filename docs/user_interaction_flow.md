## User Interaction Flow
```
USER JOURNEY: Running a Complete Pipeline Training

1. USER NAVIGATES TO PIPELINE DASHBOARD
   └──> Vue component mounts
       └──> Fetch current pipeline status (GET /pipeline/status)
           └──> Display status: "Ready", "No artifacts loaded"

2. USER CLICKS "Start Training Pipeline"
   └──> Frontend sends: POST /pipeline/run
       └──> Backend creates pipeline execution context
           └──> Starts background task
               └──> Immediately returns: 
                   {
                     "pipeline_id": "run_20260118_145230",
                     "status": "queued",
                     "message": "Pipeline queued, initializing..."
                   }

3. FRONTEND POLLS STATUS (every 2-3 seconds)
   └──> GET /pipeline/status
       └──> Returns:
           {
             "pipeline_id": "run_20260118_145230",
             "status": "running",
             "current_stage": "stage_1_preprocessing",
             "progress": {
               "stage_1_preprocessing": {
                 "status": "running",
                 "progress_percent": 45,
                 "elapsed_seconds": 120
               },
               "stage_2_embeddings": {"status": "pending"},
               "stage_3_matrices": {"status": "pending"},
               "stage_4_training": {"status": "pending"},
               "stage_5_evaluation": {"status": "pending"}
             },
             "overall_progress": 9
           }

4. DASHBOARD UPDATES IN REAL-TIME
   ├──> Progress bars animate
   ├──> Current stage highlighted
   └──> Elapsed time updates

5. STAGE COMPLETES, NEXT BEGINS
   └──> Status updates automatically
       └──> User sees: "Stage 1 ✓ → Stage 2 running..."

6. PIPELINE COMPLETES
   └──> Final status:
       {
         "pipeline_id": "run_20260118_145230",
         "status": "completed",
         "total_duration_seconds": 2847,
         "artifacts_generated": [
           "cleaned_books_data.ftr",
           "catalog_books_384.npy",
           "train_matrix.npz",
           "book_factors.npy",
           "user_factors.npy"
         ]
       }
   
7. USER CLICKS "Load Artifacts"
   └──> Backend reloads RecommendationService with new models
       └──> Frontend notifies: "Models reloaded! Ready for recommendations"

## Advanced Control Flow (Optional)
```
MANUAL STAGE EXECUTION

User wants to rerun only Stage 3 (matrices) without full pipeline:

1. USER NAVIGATES TO "Manual Stage Selection"
   └──> Sees checklist of stages with dependency indicators

2. USER SELECTS: "Run Stage 3 (Rebuild Matrices)"
   └──> Frontend warns: "Stage 1 & 2 must be completed first"
   
3. USER CLICKS "Run Stage 3 Only"
   └──> Sends: POST /pipeline/stage/stage_3_matrices
   
4. BACKEND VALIDATES PREREQUISITES
   ├──> Checks: "Do Stage 1, 2 outputs exist?"
   ├──> If yes: Executes Stage 3
   └──> If no: Returns error with instructions

5. RESULT: Fresh matrices generated
   └──> User can immediately trigger Stage 4 without waiting

## Error Handling Flow
```
SCENARIO: Stage 2 (Embeddings) fails due to out of memory

1. Backend detects error during execution
```
USER JOURNEY: Running a Complete Pipeline Training

1. USER NAVIGATES TO PIPELINE DASHBOARD
   └─> Vue component mounts
       └─> Fetch current pipeline status (GET /pipeline/status)
           └─> Display status: "Ready", "No artifacts loaded"

2. USER CLICKS "Start Training Pipeline"
   └─> Frontend sends: POST /pipeline/run
       └─> Backend creates pipeline execution context
           └─> Starts background task
               └─> Immediately returns: 
                   {
                     "pipeline_id": "run_20260118_145230",
                     "status": "queued",
                     "message": "Pipeline queued, initializing..."
                   }

3. FRONTEND POLLS STATUS (every 2-3 seconds)
   └─> GET /pipeline/status
       └─> Returns:
           {
             "pipeline_id": "run_20260118_145230",
             "status": "running",
             "current_stage": "stage_1_preprocessing",
             "progress": {
               "stage_1_preprocessing": {
                 "status": "running",
                 "progress_percent": 45,
                 "elapsed_seconds": 120
               },
               "stage_2_embeddings": {"status": "pending"},
               "stage_3_matrices": {"status": "pending"},
               "stage_4_training": {"status": "pending"},
               "stage_5_evaluation": {"status": "pending"}
             },
             "overall_progress": 9
           }

4. DASHBOARD UPDATES IN REAL-TIME
   ├─> Progress bars animate
   ├─> Current stage highlighted
   └─> Elapsed time updates

5. STAGE COMPLETES, NEXT BEGINS
   └─> Status updates automatically
       └─> User sees: "Stage 1 ✓ → Stage 2 running..."

6. PIPELINE COMPLETES
   └─> Final status:
       {
         "pipeline_id": "run_20260118_145230",
         "status": "completed",
         "total_duration_seconds": 2847,
         "artifacts_generated": [
           "cleaned_books_data.ftr",
           "catalog_books_384.npy",
           "train_matrix.npz",
           "book_factors.npy",
           "user_factors.npy"
         ]
       }
   
7. USER CLICKS "Load Artifacts"
   └─> Backend reloads RecommendationService with new models
       └─> Frontend notifies: "Models reloaded! Ready for recommendations"
```


## Advanced Control Flow (Optional)
```
MANUAL STAGE EXECUTION

User wants to rerun only Stage 3 (matrices) without full pipeline:

1. USER NAVIGATES TO "Manual Stage Selection"
   └─> Sees checklist of stages with dependency indicators
   
2. USER SELECTS: "Run Stage 3 (Rebuild Matrices)"
   └─> Frontend warns: "Stage 1 & 2 must be completed first"
   
3. USER CLICKS "Run Stage 3 Only"
   └─> Sends: POST /pipeline/stage/stage_3_matrices
   
4. BACKEND VALIDATES PREREQUISITES
   ├─> Checks: "Do Stage 1, 2 outputs exist?"
   ├─> If yes: Executes Stage 3
   └─> If no: Returns error with instructions

5. RESULT: Fresh matrices generated
   └─> User can immediately trigger Stage 4 without waiting
```

## Error Handling Flow
```
SCENARIO: Stage 2 (Embeddings) fails due to out of memory

1. Backend detects error during execution
   └─> Logs error details
       └─> Updates status: "failed"

2. Frontend detects status change
   └─> Shows error banner:
       "Stage 2 Failed: Out of Memory"
       "Suggestion: Reduce batch size"
       [Retry] [View Logs] [Reset Pipeline]

3. USER OPTIONS:
   a) [Retry] → Rerun stage with same config
   b) [View Logs] → GET /pipeline/logs/stage_2_embeddings
   c) [Reset] → DELETE /pipeline/reset
      → Start fresh from Stage 1
```

## Frontend Vue Structure
```
<!-- App.vue -->
<template>
  <div style="...">
    <h2>Conversational Book Recommender</h2>
    
    <!-- EXISTING: Recommendation section -->
    <RecommendationPanel />
    
    <!-- NEW: Pipeline Management section -->
    <div v-if="showPipelinePanel" style="margin-top: 32px;">
      <h3>🔧 Pipeline Management</h3>
      
      <PipelineDashboard 
        :pipeline-status="pipelineStatus"
        :current-stage="currentStage"
        :progress="overallProgress"
      />
      
      <PipelineControls 
        @run-full-pipeline="runFullPipeline"
        @run-stage="runStage"
        @load-artifacts="loadArtifacts"
        :disabled="isRunning"
      />
      
      <ArtifactViewer 
        :artifacts="artifacts"
        @delete="deleteArtifact"
      />
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      showPipelinePanel: false,  // Toggle with admin button
      pipelineStatus: null,
      currentStage: null,
      overallProgress: 0,
      artifacts: [],
      isRunning: false,
      pollInterval: null,
    }
  },
  
  methods: {
    async runFullPipeline() {
      this.isRunning = true
      const res = await fetch('http://localhost:8000/pipeline/run', {
        method: 'POST'
      })
      const data = await res.json()
      this.pipelineId = data.pipeline_id
      this.startPolling()
    },
    
    async runStage(stageId) {
      this.isRunning = true
      const res = await fetch(
        `http://localhost:8000/pipeline/stage/${stageId}`,
        { method: 'POST' }
      )
      const data = await res.json()
      this.startPolling()
    },
    
    startPolling() {
      this.pollInterval = setInterval(async () => {
        const res = await fetch('http://localhost:8000/pipeline/status')
        const data = await res.json()
        
        this.pipelineStatus = data.status
        this.currentStage = data.current_stage
        this.overallProgress = data.overall_progress
        this.artifacts = data.artifacts_generated || []
        
        if (data.status === 'completed' || data.status === 'failed') {
          this.isRunning = false
          clearInterval(this.pollInterval)
        }
      }, 2000)
    },
    
    async loadArtifacts() {
      const res = await fetch('http://localhost:8000/pipeline/reload-models', {
        method: 'POST'
      })
      const data = await res.json()
      alert(`Models loaded: ${data.message}`)
    }
  }
}
</script>
```

## Backend API Endpoints (main.py additions)
```
from server.pipeline_service import PipelineService

pipeline_service = PipelineService()

@app.post("/pipeline/run")
async def run_full_pipeline(background_tasks):
    """Trigger full pipeline execution"""
    pipeline_id = pipeline_service.create_pipeline_run()
    
    # Run in background without blocking request
    background_tasks.add_task(
        pipeline_service.execute_full_pipeline,
        pipeline_id
    )
    
    return {
        "pipeline_id": pipeline_id,
        "status": "queued",
        "message": "Pipeline queued for execution"
    }

@app.post("/pipeline/stage/{stage_id}")
async def run_stage(stage_id: str, background_tasks):
    """Trigger specific stage"""
    result = pipeline_service.validate_stage(stage_id)
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["reason"])
    
    pipeline_id = pipeline_service.create_pipeline_run()
    background_tasks.add_task(
        pipeline_service.execute_stage,
        pipeline_id,
        stage_id
    )
    
    return {
        "pipeline_id": pipeline_id,
        "stage_id": stage_id,
        "status": "queued"
    }

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status"""
    return pipeline_service.get_status()

@app.get("/pipeline/logs/{stage_id}")
async def get_stage_logs(stage_id: str):
    """Get detailed logs for a specific stage"""
    return pipeline_service.get_stage_logs(stage_id)

@app.post("/pipeline/reload-models")
async def reload_models():
    """Reload recommendation models from latest artifacts"""
    service.reload_from_artifacts()
    return {"status": "ok", "message": "Models reloaded successfully"}

@app.delete("/pipeline/reset")
async def reset_pipeline():
    """Clean up pipeline artifacts and state"""
    pipeline_service.reset()
    return {"status": "ok", "message": "Pipeline reset"}
```

## Pipeline Service (server/pipeline_service.py)
```
class PipelineService:
    def __init__(self):
        self.state_file = DATA_DIR / "pipeline_state.json"
        self.current_run = None
    
    def create_pipeline_run(self) -> str:
        """Create new pipeline execution context"""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run = {
            "id": run_id,
            "status": "queued",
            "stages": {}
        }
        return run_id
    
    def execute_full_pipeline(self, pipeline_id: str):
        """Execute all stages with status tracking"""
        # Implemented by calling ml_pipeline/pipeline_executor.py
    
    def get_status(self) -> dict:
        """Return current pipeline state"""
        # Read from state_file, calculate progress
    
    def save_state(self):
        """Persist state to JSON file"""
        # Save self.current_run to state_file
```