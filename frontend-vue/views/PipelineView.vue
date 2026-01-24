<template>
    <main class="main-content">
        <div class="pipeline-container">
            <div v-if="!running" class="empty-state">
                <div class="empty-card">
                    <div class="empty-icon">⚙️</div>
                    <h2 class="empty-title">No Data Available</h2>
                    <p class="empty-description">
                        To get started with personalized recommendations, we need to run the ML pipeline to generate
                        recommendation artifacts.
                    </p>
                    <button class="btn btn-primary" @click="runPipeline">
                        Run Pipeline
                    </button>
                </div>
            </div>

            <div v-else class="pipeline-progress-container">
                <div class="progress-card">
                    <div class="progress-header">
                        <h3 class="progress-title">Running ML Pipeline</h3>
                        <div class="spinner-inline"></div>
                    </div>

                    <div class="progress-bar-wrapper">
                        <div class="progress-bar">
                            <div class="progress-fill" :style="{ width: progress + '%' }"></div>
                        </div>
                        <span class="progress-percent">{{ progress }}%</span>
                    </div>

                    <p class="progress-status">{{ status }}</p>

                    <div class="progress-info">
                        <p>This may take a few minutes. Please keep this window open.</p>
                    </div>
                </div>
            </div>
        </div>
    </main>
</template>

<script>
import { runPipeline, fetchPipelineStatus } from '../services/api.js'

export default {
    emits: ['pipeline-complete'],

    data() {
        return {
            running: false,
            progress: 0,
            status: 'Initializing...',
            pollingInterval: null
        }
    },

    methods: {
        async runPipeline() {
            try {
                this.running = true
                this.progress = 0
                this.status = 'Initializing...'

                await runPipeline()
                this.startPolling()
            } catch (error) {
                console.error('Error starting pipeline:', error)
                this.running = false
                this.status = 'Error: Failed to start pipeline'
            }
        },

        startPolling() {
            if (this.pollingInterval) clearInterval(this.pollingInterval)

            this.pollingInterval = setInterval(async () => {
                try {
                    const data = await fetchPipelineStatus()

                    if (data.status === 'running') {
                        this.progress = Math.min(data.overall_progress || 0, 99)
                        const stages = data.progress || {}
                        const runningStage = Object.entries(stages).find(
                            ([_, info]) => info.status === 'running'
                        )
                        if (runningStage) {
                            const stageName = runningStage[0].replace(/_/g, ' ')
                            this.status = `Running: ${stageName}`
                        }
                    } else if (data.status === 'completed') {
                        this.progress = 100
                        this.status = 'Pipeline completed successfully!'
                        if (this.pollingInterval) clearInterval(this.pollingInterval)
                        setTimeout(() => {
                            this.$emit('pipeline-complete')
                        }, 1500)
                    } else if (data.status === 'failed') {
                        this.running = false
                        if (this.pollingInterval) clearInterval(this.pollingInterval)
                        this.status = `Error: ${data.error_message || 'Unknown error'}`
                    }
                } catch (error) {
                    console.error('Error polling pipeline status:', error)
                }
            }, 2000)
        }
    },

    beforeUnmount() {
        if (this.pollingInterval) clearInterval(this.pollingInterval)
    }
}
</script>
