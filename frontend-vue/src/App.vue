<template>
    <div class="app-container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <div class="header-title">
                    <h1>📚 Book Explorer</h1>
                    <p class="subtitle">Discover your next favorite book</p>
                </div>
            </div>
        </header>

        <!-- Toast Notification -->
        <Toast :message="toast.message" :type="toast.type" :icon="toast.icon" :show="toast.show" />

        <!-- Current View -->
        <StartView v-if="!initialized" @data-ready="onDataReady" />
        <PipelineView v-else-if="!dataReady" @pipeline-complete="onPipelineComplete" />
        <RecommendationsView v-else ref="recommendationsView" />
    </div>
</template>

<script>
import './styles.css'
import Toast from '../components/Toast.vue'
import StartView from './StartView.vue'
import PipelineView from '../views/PipelineView.vue'
import RecommendationsView from '../views/RecommendationsView.vue'
import { checkRecommendationStatus } from '../services/api.js'

export default {
    components: {
        Toast,
        StartView,
        PipelineView,
        RecommendationsView
    },

    data() {
        return {
            initialized: false,
            dataReady: false,
            toast: {
                show: false,
                message: '',
                type: 'success',
                icon: '✓'
            }
        }
    },

    methods: {
        async checkDataStatus() {
            try {
                const data = await checkRecommendationStatus()
                this.dataReady = !!data.ready
            } catch (error) {
                console.error('Error checking data status:', error)
                this.dataReady = false
            }
            this.initialized = true
        },

        onDataReady() {
            this.dataReady = true
            this.showToast('Data is ready! Loading recommendations...', 'success', '✓')
        },

        onPipelineComplete() {
            this.dataReady = true
            this.showToast('Pipeline completed successfully!', 'success', '✓')
        },

        showToast(message, type = 'success', icon = '✓') {
            this.toast = { show: true, message, type, icon }
            setTimeout(() => {
                this.toast.show = false
            }, 4000)
        }
    },

    mounted() {
        this.checkDataStatus()
    }
}
</script>