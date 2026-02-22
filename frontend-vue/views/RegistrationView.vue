<template>
  <main class="main-content register-view">
    <div class="register-container">
      <div class="register-card">
        <div class="register-icon">✨</div>
        <h2 class="register-title">Create Account</h2>
        <p class="register-subtitle">Join Book Explorer to get personalized recommendations</p>

        <form @submit.prevent="handleRegister" class="register-form">
          <!-- User Name Input -->
          <div class="form-group">
            <label for="name" class="form-label">Display Name</label>
            <input id="name" v-model="form.name" type="text" class="form-input" placeholder="e.g., Alice Smith"
              :disabled="loading" @input="clearErrors" />
            <span v-if="errors.name" class="error-message">{{ errors.name }}</span>
          </div>

          <!-- User ID Input -->
          <div class="form-group">
            <label for="user-id" class="form-label">User ID</label>
            <input id="user-id" v-model="form.userId" type="text" class="form-input" placeholder="e.g., alice_smith_001"
              :disabled="loading" @input="clearErrors" @blur="checkUserIdAvailability" />
            <span v-if="errors.userId" class="error-message">{{ errors.userId }}</span>
            <span v-if="userIdChecking" class="info-message">Checking availability...</span>
            <span v-if="userIdAvailable && form.userId" class="success-message">
              ✓ User ID is available
            </span>
          </div>

          <button type="submit" class="btn-primary btn-register" :disabled="loading || !isFormValid">
            <span v-if="!loading">Create Account</span>
            <span v-else class="button-spinner"></span>
          </button>
        </form>

        <div class="register-divider">
          <span>Already have an account?</span>
        </div>

        <button type="button" class="btn-secondary btn-login" @click="$emit('switch-to-login')" :disabled="loading">
          Sign In
        </button>
      </div>
    </div>
  </main>
</template>

<script>
import { checkUserIdExists, createUser } from '../services/api.js';

export default {
  emits: ['registration-success', 'switch-to-login'],

  data() {
    return {
      form: {
        name: '',
        userId: ''
      },
      errors: {
        name: '',
        userId: ''
      },
      loading: false,
      userIdChecking: false,
      userIdAvailable: false
    }
  },

  computed: {
    isFormValid() {
      return (
        this.form.name.trim().length > 0 &&
        this.form.userId.trim().length > 0 &&
        !this.errors.name &&
        !this.errors.userId &&
        this.userIdAvailable
      )
    }
  },

  methods: {
    clearErrors() {
      this.errors = { name: '', userId: '' }
      this.userIdAvailable = false
    },

    async checkUserIdAvailability() {
      if (!this.form.userId.trim()) {
        this.userIdAvailable = false
        return
      }

      this.userIdChecking = true
      this.userIdAvailable = false

      try {
        const exists = await checkUserIdExists(this.form.userId)
        if (exists) {
          this.errors.userId = 'User ID already taken, try another'
        } else {
          this.errors.userId = ''
          this.userIdAvailable = true
        }
      } catch (err) {
        console.error('Error checking user ID:', err)
        this.errors.userId = 'Could not verify user ID. Please try again.'
      } finally {
        this.userIdChecking = false
      }
    },

    async handleRegister() {
      // Validate name
      if (!this.form.name.trim()) {
        this.errors.name = 'Display name is required'
        return
      }

      // Validate user ID
      if (!this.form.userId.trim()) {
        this.errors.userId = 'User ID is required'
        return
      }

      if (!this.userIdAvailable) {
        if (!this.errors.userId) {
          this.errors.userId = 'Please verify your user ID is available'
        }
        return
      }

      this.loading = true

      try {
        const response = await createUser({
          name: this.form.name.trim(),
          user_id: this.form.userId.trim()
        })

        console.log('Registration response:', response)
        if (response.first_login === undefined) {
            console.warn('first_login missing from registration response!')
            response.first_login = true  // Fallback: new users always need genres
        }
        
        // Emit the full response object with user_id and name
        this.$emit('registration-success', response)
      } catch (err) {
        this.errors.userId = err.message || 'Registration failed. Please try again.'
        console.error('Registration error:', err)
      } finally {
        this.loading = false
      }
    }
  }
}
</script>

<style scoped>
.register-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, var(--primary-pink-light) 0%, var(--login-bg) 100%);
}

.register-container {
  width: 100%;
  max-width: 400px;
  padding: 20px;
}

.register-card {
  background: var(--bg-primary);
  border-radius: 16px;
  padding: 40px;
  box-shadow: var(--shadow-lg);
  animation: slideUp 0.6s ease-out;
}

.register-icon {
  font-size: 48px;
  text-align: center;
  margin-bottom: 20px;
}

.register-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  text-align: center;
  margin-bottom: 8px;
}

.register-subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  text-align: center;
  margin-bottom: 30px;
}

.register-form {
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

.info-message {
  font-size: 13px;
  color: var(--text-tertiary);
  margin-top: -4px;
}

.success-message {
  font-size: 13px;
  color: var(--success-color);
  margin-top: -4px;
  font-weight: 600;
}

.btn-register {
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

.btn-register:hover:not(:disabled) {
  background: var(--primary-pink-dark);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.btn-register:disabled {
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

.register-divider {
  text-align: center;
  margin: 20px 0;
  font-size: 13px;
  color: var(--text-tertiary);
}

.btn-login {
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

.btn-login:hover:not(:disabled) {
  background: var(--primary-pink-light);
  transform: translateY(-2px);
}

.btn-login:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
