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
