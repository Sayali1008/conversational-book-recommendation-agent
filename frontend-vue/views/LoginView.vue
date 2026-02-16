<template>
  <main class="main-content login-view">
    <div class="login-container">
      <div class="login-card">
        <div class="login-icon">📚</div>
        <h2 class="login-title">Book Explorer</h2>
        <p class="login-subtitle">Sign in to get personalized recommendations</p>

        <form @submit.prevent="handleLogin" class="login-form">
          <div class="form-group">
            <label for="user-id" class="form-label">User ID</label>
            <input id="user-id" v-model="form.userId" type="text" class="form-input" placeholder="Enter your user ID"
              :disabled="loading" @input="clearError" />
            <span v-if="error" class="error-message">{{ error }}</span>
          </div>

          <button type="submit" class="btn-primary btn-login" :disabled="loading || !form.userId.trim()">
            <span v-if="!loading">Sign In</span>
            <span v-else class="button-spinner"></span>
          </button>
        </form>

        <div class="login-divider">
          <span>Don't have an account?</span>
        </div>

        <button type="button" class="btn-secondary btn-register" @click="$emit('switch-to-register')"
          :disabled="loading">
          Create Account
        </button>
      </div>
    </div>
  </main>
</template>

<script>
import { loginUser } from '../services/api.js'

export default {
  emits: ['login-success', 'switch-to-register'],

  data() {
    return {
      form: {
        userId: ''
      },
      loading: false,
      error: ''
    }
  },

  methods: {
    async handleLogin() {
      this.error = ''

      // Validation
      if (!this.form.userId.trim()) {
        this.error = 'User ID is required'
        return
      }

      this.loading = true

      try {
        const response = await loginUser(this.form.userId)
        // Emit the full response object with user_id and name
        this.$emit('login-success', response)
      } catch (err) {
        this.error = err.message || 'Login failed. Please try again.'
        console.error('Login error:', err)
      } finally {
        this.loading = false
      }
    },

    clearError() {
      this.error = ''
    }
  }
}
</script>

<style scoped>
.login-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, var(--primary-pink-light) 0%, var(--login-bg) 100%);
}

.login-container {
  width: 100%;
  max-width: 400px;
  padding: 20px;
}

.login-card {
  background: var(--bg-primary);
  border-radius: 16px;
  padding: 40px;
  box-shadow: var(--shadow-lg);
  animation: slideUp 0.6s ease-out;
}

.login-icon {
  font-size: 48px;
  text-align: center;
  margin-bottom: 20px;
}

.login-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  text-align: center;
  margin-bottom: 8px;
}

.login-subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  text-align: center;
  margin-bottom: 30px;
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-bottom: 25px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.form-input {
  padding: 12px 16px;
  border: 2px solid var(--border-color);
  border-radius: 8px;
  font-size: 14px;
  color: var(--text-primary);
  transition: all 0.3s ease;
  background: var(--bg-secondary);
}

.form-input:focus {
  outline: none;
  border-color: var(--primary-pink);
  background: var(--bg-primary);
  box-shadow: 0 0 0 3px rgba(255, 182, 193, 0.2);
}

.form-input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  font-size: 13px;
  color: var(--error-color);
  margin-top: -4px;
}

.btn-login {
  width: 100%;
  padding: 12px 16px;
  font-size: 16px;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: var(--primary-pink);
  color: white;
  margin-top: 10px;
}

.btn-login:hover:not(:disabled) {
  background: var(--primary-pink-dark);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.btn-login:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.button-spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.login-divider {
  text-align: center;
  margin: 20px 0;
  font-size: 13px;
  color: var(--text-tertiary);
}

.btn-register {
  width: 100%;
  padding: 12px 16px;
  font-size: 16px;
  font-weight: 600;
  border: 2px solid var(--primary-pink);
  border-radius: 8px;
  background: transparent;
  color: var(--primary-pink-dark);
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-register:hover:not(:disabled) {
  background: var(--primary-pink-light);
  transform: translateY(-2px);
}

.btn-register:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
