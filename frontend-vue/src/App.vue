<template>
    <div class="app-container">
        <!-- Header (shown when authenticated) -->
        <header v-if="authenticated" class="app-header">
            <div class="header-left">
                <h1 class="header-title">📚 Book Explorer</h1>
                <p class="header-subtitle">Discover your next favorite book</p>
            </div>

            <div class="header-right">
                <!-- Add Book Button -->
                <button v-if="dataReady" class="btn-icon btn-add-book" @click="showAddBookModal = true"
                    title="Add a new book">
                    ➕ Add Book
                </button>

                <!-- Profile Menu -->
                <div class="profile-container">
                    <button class="profile-icon" @click="showProfileMenu = !showProfileMenu" :title="userName">
                        {{ userName.charAt(0).toUpperCase() }}
                    </button>
                    <div v-if="showProfileMenu" class="profile-menu" @click="showProfileMenu = false">
                        <div class="profile-menu-header">
                            <div class="profile-name">{{ userName }}</div>
                            <div class="profile-id">@{{ userId }}</div>
                        </div>
                        <button class="profile-logout" @click="logout">
                            🚪 Logout
                        </button>
                    </div>
                </div>
            </div>
        </header>

        <!-- Toast Notification -->
        <Toast :message="toast.message" :type="toast.type" :icon="toast.icon" :show="toast.show" />

        <!-- Authentication Views -->
        <LoginView v-if="!authenticated && authView === 'login'" @login-success="onLoginSuccess"
            @switch-to-register="authView = 'register'" />
        <RegistrationView v-else-if="!authenticated && authView === 'register'" @registration-success="onLoginSuccess"
            @switch-to-login="authView = 'login'" />

        <!-- Main App Views -->
        <div v-else-if="authenticated" class="app-main">
            <!-- Genre Selection for first-time users -->
            <GenreSelectView v-if="showGenreSelect" @genres-saved="onGenresSaved" />

            <!-- Loading state -->
            <div v-else-if="dataReady === false && !initialized" class="loading-view">
                <div class="loading-spinner"></div>
                <p>Checking recommendation data...</p>
            </div>

            <!-- Pipeline view - shown when data not ready or explicitly triggered -->
            <PipelineView v-else-if="showPipelineView" @pipeline-complete="onPipelineComplete"
                @cancel="showPipelineView = false" />

            <!-- Main recommendations view -->
            <RecommendationsView v-else-if="dataReady" @show-book-details="showBookDetails" />
        </div>

        <!-- Book Details Modal -->
        <BookDetailsModal v-if="selectedBook" :book="selectedBook" @close="selectedBook = null" />

        <!-- Add Book Modal -->
        <AddBookModal v-if="showAddBookModal" @book-added="onBookAdded" @close="showAddBookModal = false" />
    </div>
</template>

<script>
import './styles.css'
import Toast from '../components/Toast.vue'
import LoginView from '../views/LoginView.vue'
import RegistrationView from '../views/RegistrationView.vue'
import GenreSelectView from '../views/GenreSelectView.vue'
import PipelineView from '../views/PipelineView.vue'
import RecommendationsView from '../views/RecommendationsView.vue'
import BookDetailsModal from '../components/BookDetailsModal.vue'
import AddBookModal from '../components/AddBookModal.vue'
import { checkRecommendationStatus, fetchBook } from '../services/api.js'

export default {
    components: {
        Toast,
        LoginView,
        RegistrationView,
        GenreSelectView,
        PipelineView,
        RecommendationsView,
        BookDetailsModal,
        AddBookModal
    },

    data() {
        return {
            authenticated: false,
            userId: '',
            userName: '',
            authView: 'login',
            initialized: false,
            dataReady: false,
            showProfileMenu: false,
            showAddBookModal: false,
            showPipelineView: false,
            selectedBook: null,
            showGenreSelect: false,
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
            // Automatically show pipeline if no recommendations available
            if (!this.dataReady) {
                this.showPipelineView = true
            }
        },

        onLoginSuccess(userData) {
            this.authenticated = true
            this.userId = userData.user_id
            this.userName = userData.name

            console.log('onLoginSuccess received userData:', userData)
            console.log('first_login value:', userData.first_login, 'type:', typeof userData.first_login)
            
            sessionStorage.setItem('userId', this.userId)
            sessionStorage.setItem('userName', this.userName)
            this.showToast(`Welcome, ${this.userName}!`, 'success', '✓')

            // Show genre selection for first-time users
            this.showGenreSelect = userData.first_login === true

            // For returning users, immediately check data status
            if (!userData.first_login) {
                this.checkDataStatus()
            }
        },

        onPipelineComplete() {
            this.dataReady = true
            this.showPipelineView = false
            this.showToast('Pipeline completed! Ready for recommendations.', 'success', '✓')
        },

        onBookAdded(book) {
            this.showAddBookModal = false
            this.showToast(`Book "${book.title}" added successfully!`, 'success', '✓')
        },

        onGenresSaved() {
            this.showGenreSelect = false
            this.checkDataStatus()
        },

        showBookDetails(book) {
            // Fetch full book details including genres from the backend
            fetchBook(book.book_id)
                .then(fullBook => {
                    // Preserve the score from the original book object
                    this.selectedBook = { ...fullBook, score: book.score }
                })
                .catch(err => {
                    console.error('Error fetching book details:', err)
                    // Fall back to showing the basic book info
                    this.selectedBook = book
                })
        },

        logout() {
            sessionStorage.removeItem('userId')
            sessionStorage.removeItem('userName')
            this.authenticated = false
            this.userId = ''
            this.userName = ''
            this.showProfileMenu = false
            this.initialized = false
            this.dataReady = false
            this.authView = 'login'
            this.showToast('You have been logged out', 'info', '👋')
        },

        showToast(message, type = 'success', icon = '✓') {
            this.toast = { show: true, message, type, icon }
            setTimeout(() => {
                this.toast.show = false
            }, 4000)
        }
    },

    mounted() {
        const storedUserId = sessionStorage.getItem('userId')
        const storedUserName = sessionStorage.getItem('userName')
        if (storedUserId) {
            this.authenticated = true
            this.userId = storedUserId
            this.userName = storedUserName || storedUserId
            this.checkDataStatus()
        }

        // Close profile menu on click outside
        this.handleClickOutside = (event) => {
            const profile = document.querySelector('.profile-container')
            if (profile && !profile.contains(event.target)) {
                this.showProfileMenu = false
            }
        }
        document.addEventListener('click', this.handleClickOutside)
    },

    beforeUnmount() {
        if (this.handleClickOutside) {
            document.removeEventListener('click', this.handleClickOutside)
        }
    }
}
</script>

<style scoped>
.app-container {
    min-height: 100vh;
    background-color: var(--bg-secondary);
    display: flex;
    flex-direction: column;
}

/* ===== HEADER ===== */
.app-header {
    background-color: var(--bg-primary);
    border-bottom: 2px solid var(--primary-purple-light);
    padding: 20px 100px;
    display: flex;
    justify-content: space-between;
    height: 17vh;
    align-items: center;
    box-shadow: var(--shadow-md);
}

.header-left {
    flex: 1;
}

.header-title {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-secondary);
    margin: 0;
    padding: 0;
}

.header-subtitle {
    font-size: 14px;
    color: var(--text-tertiary);
    margin: 4px 0 0 0;
    padding: 0;
}

.header-right {
    display: flex;
    gap: 20px;
    align-items: center;
}

.btn-add-book {
    background-color: var(--primary-pink);
    color: var(--text-primary);
    padding: 10px 16px;
    border-radius: 8px;
    border: none;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-add-book:hover {
    background-color: var(--primary-pink-dark);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

/* ===== PROFILE MENU ===== */
.profile-container {
    position: relative;
}

.profile-icon {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background-color: var(--primary-purple);
    color: white;
    border: none;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
}

.profile-icon:hover {
    background-color: var(--primary-purple-dark);
    transform: scale(1.05);
}

.profile-menu {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 8px;
    background-color: var(--bg-primary);
    border-radius: 8px;
    box-shadow: var(--shadow-lg);
    min-width: 200px;
    z-index: 1000;
    overflow: hidden;
}

.profile-menu-header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
}

.profile-name {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 14px;
}

.profile-id {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
}

.profile-logout {
    width: 100%;
    padding: 12px 16px;
    background: none;
    border: none;
    color: var(--text-primary);
    font-size: 14px;
    cursor: pointer;
    text-align: left;
    transition: background-color 0.2s ease;
}

.profile-logout:hover {
    background-color: var(--bg-secondary);
}

/* ===== MAIN CONTENT ===== */
.app-main {
    flex: 1;
    overflow: auto;
    padding: 40px;
}

.loading-view {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 400px;
}

.loading-spinner {
    width: 50px;
    height: 50px;
    border: 4px solid var(--border-color);
    border-top: 4px solid var(--primary-purple);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

.btn-primary {
    background-color: var(--primary-purple);
    color: white;
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-primary:hover {
    background-color: var(--primary-purple-dark);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.btn-icon {
    background-color: transparent;
    border: none;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.2s ease;
}
</style>
