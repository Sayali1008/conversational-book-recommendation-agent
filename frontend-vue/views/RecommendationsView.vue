<template>
    <main class="main-content">
        <!-- Input Form Panel -->
        <section class="search-panel">
            <RecommendationForm :k="k" :userId="userId" :seedBookIds="seedBookIds" :loading="loading"
                @submit="handleSubmit" @update:k="k = $event" @update:userId="userId = $event"
                @update:seedBookIds="seedBookIds = $event" />
        </section>

        <!-- Strategy Display -->
        <div v-if="strategy" class="strategy-badge">
            <span class="strategy-label">Strategy:</span>
            <span class="strategy-value">{{ strategy }}</span>
        </div>

        <!-- Recommendations Section -->
        <section class="recommendations-section">
            <!-- Empty State -->
            <div v-if="!recommendations.length && !loading" class="empty-state">
                <div class="empty-icon">📖</div>
                <h3>No recommendations yet</h3>
                <p>Click "Get Recommendations" to get started</p>
            </div>

            <!-- Recommendations Grid -->
            <div v-else-if="recommendations.length" class="recommendations-grid">
                <div v-for="(rec, idx) in recommendations" :key="rec.book_id" class="book-card"
                    :style="{ animationDelay: `${idx * 50}ms` }">
                    <!-- Book Cover -->
                    <div class="book-cover">
                        <button class="cover-button" @click="openBookModal(rec.book_id)" :disabled="loading">
                            <div class="cover-placeholder">
                                <span class="cover-icon">📕</span>
                            </div>
                        </button>
                    </div>

                    <!-- Book Info -->
                    <div class="book-info">
                        <button class="book-title-button" @click="openBookModal(rec.book_id)" :disabled="loading">
                            {{ rec.title }}
                        </button>
                        <div v-if="rec.authors && rec.authors.length" class="book-authors">
                            {{ rec.authors.join(', ') }}
                        </div>

                        <div class="book-meta">
                            <span class="meta-badge score">⭐ {{ rec.score.toFixed(2) }}</span>
                            <span class="meta-badge source">{{ rec.source }}</span>
                            <span class="meta-badge source">ID: {{ rec.book_id }}</span>
                        </div>
                    </div>

                    <!-- Action Buttons -->
                    <div class="book-actions">
                        <button class="action-btn dislike" @click="handleSwipe(rec, 'dislike')" :disabled="loading"
                            title="I don't like this">
                            <span>👎</span>
                        </button>
                        <button class="action-btn like" @click="handleSwipe(rec, 'like')" :disabled="loading"
                            title="I like this!">
                            <span>👍</span>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Loading State -->
            <div v-if="loading" class="loading-state">
                <div v-for="i in k" :key="i" class="loading-skeleton">
                    <div class="skeleton-cover"></div>
                    <div class="skeleton-content">
                        <div class="skeleton-line"></div>
                        <div class="skeleton-line short"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Book Details Modal -->
        <BookModal v-if="selectedBook" :book="selectedBook" :loading="loading" @close="closeBookModal"
            @like="handleModalSwipe('like')" @dislike="handleModalSwipe('dislike')" />
    </main>
</template>

<script>
import RecommendationForm from '../components/RecommendationForm.vue'
import RecommendationGrid from '../components/RecommendationGrid.vue'
import BookModal from '../components/BookModal.vue'
import { fetchRecommendations, fetchBook, swipe } from '../services/api.js'

export default {
    components: {
        RecommendationForm,
        RecommendationGrid,
        BookModal
    },

    data() {
        return {
            userId: '',
            seedBookIds: '',
            k: 3,
            recommendations: [],
            strategy: '',
            loading: false,
            selectedBook: null,
            selectedBookId: null
        }
    },

    methods: {
        async handleSubmit() {
            if (this.loading) return
            this.loading = true

            try {
                const params = {
                    k: String(this.k)
                }
                if (this.userId) params.user_id = this.userId
                if (this.seedBookIds) params.seed_book_ids = this.seedBookIds

                const data = await fetchRecommendations(params)
                this.recommendations = data.recommendations || []
                this.strategy = data.strategy || ''
            } catch (error) {
                console.error('Error fetching recommendations:', error)
                alert('Failed to fetch recommendations. Please try again.')
            } finally {
                this.loading = false
            }
        },

        async openBookModal(bookId) {
            if (this.loading) return

            this.selectedBookId = bookId
            this.selectedBook = null

            try {
                const data = await fetchBook(bookId)
                this.selectedBook = data
            } catch (error) {
                console.error('Error fetching book details:', error)
                alert('Failed to load book details')
                this.selectedBookId = null
            }
        },

        closeBookModal() {
            this.selectedBook = null
            this.selectedBookId = null
        },

        async handleSwipe(rec, action) {
            if (this.loading) return
            this.loading = true

            try {
                const payload = {
                    user_id: this.userId || 'anonymous',
                    book_id: rec.book_id,
                    action,
                    k: this.k
                }

                const data = await swipe(payload)
                if (data.next_recommendations && data.next_recommendations.length > 0) {
                    this.recommendations = data.next_recommendations
                }
            } catch (error) {
                console.error('Error on swipe:', error)
                alert('Failed to process your action. Please try again.')
            } finally {
                this.loading = false
            }
        },

        async handleModalSwipe(action) {
            if (!this.selectedBook) return
            this.closeBookModal()
            const rec = this.recommendations.find(r => r.book_id === this.selectedBookId)
            if (rec) {
                await this.handleSwipe(rec, action)
            }
        }
    }
}
</script>
