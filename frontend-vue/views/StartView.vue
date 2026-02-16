<template>
  <main class="main-content start-view">
    <div class="start-container">
      <div class="start-card">
        <div class="start-icon">📖</div>
        <h2 class="start-title">Welcome to Book Explorer</h2>
        <p class="start-description">
          Get personalized book recommendations based on your preferences.
        </p>
        <p class="start-message">
          We're checking if recommendation data is available...
        </p>
        <div class="start-loader">
          <div class="spinner"></div>
        </div>
      </div>
    </div>
  </main>
</template>

<script>
export default {
  emits: ['data-ready'],

  data() {
    return {
      apiBase: 'http://localhost:8000',
      checking: true
    }
  },

  async mounted() {
    // Give parent time to finish mounting
    await this.$nextTick()
    // The parent (App.vue) will check status and emit data-ready
  }
}
</script>

<style scoped>
.start-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
}

.start-container {
  width: 100%;
  max-width: 500px;
  padding: 20px;
  display: flex;
  justify-content: center;
}

.start-card {
  background: var(--bg-primary);
  border-radius: 16px;
  padding: 40px;
  box-shadow: var(--shadow-lg);
  text-align: center;
  animation: slideUp 0.6s ease-out;
}

.start-icon {
  font-size: 64px;
  margin-bottom: 20px;
  display: block;
}

.start-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 16px;
}

.start-description {
  font-size: 16px;
  color: var(--text-secondary);
  margin-bottom: 12px;
  line-height: 1.5;
}

.start-message {
  font-size: 14px;
  color: var(--text-tertiary);
  margin-bottom: 24px;
}

.start-loader {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid var(--border-color);
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
