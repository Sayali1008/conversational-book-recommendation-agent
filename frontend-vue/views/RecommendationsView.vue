<template>
    <div class="recommendations-container">
        <!-- Loading State -->
        <div v-if="loading" class="empty-state">
            <div class="spinner"></div>
            <p>Loading recommendations...</p>
        </div>

        <!-- No More Recommendations -->
        <div v-else-if="!currentCard" class="empty-state">
            <div class="empty-icon">📭</div>
            <p>No more recommendations</p>
            <button class="btn-primary" @click="loadRecommendations">
                🔄 Load More
            </button>
        </div>

        <!-- Book Card -->
        <div v-else class="card-wrapper">
            <!-- Left Arrow (Dislike) -->
            <button class="arrow-btn" @click="handleSwipe('dislike')"
                title="Dislike (Left Arrow or Swipe Left)" :disabled="loading">
                👎
            </button>

            <!-- Card -->
            <div class="book-card" :style="cardStyle" @mousedown="startDrag" @touchstart="startDrag"
                ref="cardElement">
                <div class="card-image">
                    <div class="image-placeholder">📖</div>
                </div>

                <div class="card-info">
                    <h2 class="card-title">{{ currentCard.title }}</h2>
                    <p class="card-authors">{{ formattedAuthors }}</p>
                    <div class="card-score">
                        ⭐ {{ (currentCard.score * 100).toFixed(1) }}% Match
                    </div>
                </div>

                <!-- Visual feedback for drag -->
                <div v-if="isDragging" class="drag-overlay">
                    <div v-if="dragX < 0" class="swipe-indicator dislike">
                        👎 DISLIKE
                    </div>
                    <div v-else-if="dragX > 0" class="swipe-indicator like">
                        👍 LIKE
                    </div>
                </div>
            </div>

            <!-- Right Arrow (Like) -->
            <button class="arrow-btn" @click="handleSwipe('like')" title="Like (Right Arrow or Swipe Right)"
                :disabled="loading">
                👍
            </button>
        </div>

        <!-- Progress Indicator -->
        <!-- <div v-if="totalCards > 0" class="progress-section">
            <div class="progress-text">
                {{ cardIndex + 1 }} of {{ totalCards }} recommendations
            </div>
            <div class="progress-bar">
                <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
            </div>
        </div> -->

        <!-- Instructions -->
        <!-- <div class="instructions">
            <span>👈 Dislike</span>
            <span>Click card for details</span>
            <span>Like 👉</span>
        </div> -->
    </div>
</template>

<script>
import { fetchRecommendations, swipe } from '../services/api.js'

export default {
    emits: ['show-book-details'],

    data() {
        return {
            recommendations: [],
            cardIndex: 0,
            loading: false,
            isDragging: false,
            dragX: 0,
            dragY: 0,
            startX: 0,
            startY: 0
        }
    },

    computed: {
        currentCard() {
            return this.recommendations[this.cardIndex] || null
        },

        totalCards() {
            return this.recommendations.length
        },

        progressPercent() {
            if (this.totalCards === 0) return 0
            return ((this.cardIndex + 1) / this.totalCards) * 100
        },

        formattedAuthors() {
            if (!this.currentCard || !this.currentCard.authors) return ''
            const authors = typeof this.currentCard.authors === 'string'
                ? this.currentCard.authors.split(',').map(a => a.trim())
                : Array.isArray(this.currentCard.authors) ? this.currentCard.authors : []
            return authors.sort().join(', ')
        },

        cardStyle() {
            if (!this.isDragging) {
                return {
                    transform: 'translateX(0) rotateZ(0deg)',
                    opacity: 1
                }
            }

            const distance = Math.abs(this.dragX)
            const maxDistance = 150
            const rotationFactor = Math.min(distance / maxDistance, 1)
            const rotation = (this.dragX / Math.abs(this.dragX)) * rotationFactor * 15

            return {
                transform: `translateX(${this.dragX}px) rotateZ(${rotation}deg)`,
                opacity: 1 - (distance / 300)
            }
        }
    },

    methods: {
        async loadRecommendations() {
            this.loading = true
            try {
                const userId = sessionStorage.getItem('userId')
                const response = await fetchRecommendations({ user_id: userId, k: 10 })
                this.recommendations = response.recommendations || []
                this.cardIndex = 0
            } catch (error) {
                console.error('Error loading recommendations:', error)
                alert('Failed to load recommendations')
            } finally {
                this.loading = false
            }
        },

        async handleSwipe(action) {
            if (!this.currentCard || this.loading) return

            this.loading = true
            try {
                const userId = sessionStorage.getItem('userId')
                const response = await swipe({
                    user_id: userId,
                    book_id: this.currentCard.book_id,
                    action: action,
                    k: 10
                })

                // Reset drag state
                this.dragX = 0
                this.dragY = 0
                this.isDragging = false

                // Move to next card
                if (this.cardIndex < this.recommendations.length - 1) {
                    this.cardIndex++
                } else if (response.next_recommendations && response.next_recommendations.length > 0) {
                    // Load next batch if available
                    this.recommendations = response.next_recommendations
                    this.cardIndex = 0
                } else {
                    // No more cards
                    this.cardIndex = this.recommendations.length
                }
            } catch (error) {
                console.error(`Error swiping ${action}:`, error)
                alert(`Failed to record ${action}. Please try again.`)
            } finally {
                this.loading = false
            }
        },

        showDetails() {
            if (this.currentCard) {
                this.$emit('show-book-details', this.currentCard)
            }
        },

        startDrag(event) {
            if (this.loading) return
            this.isDragging = true
            this.startX = event.type.includes('mouse') ? event.clientX : event.touches[0].clientX
            this.startY = event.type.includes('mouse') ? event.clientY : event.touches[0].clientY

            const moveHandler = event.type.includes('mouse')
                ? this.handleMouseMove.bind(this)
                : this.handleTouchMove.bind(this)
            const endHandler = event.type.includes('mouse')
                ? this.handleMouseUp.bind(this)
                : this.handleTouchEnd.bind(this)

            const moveEvent = event.type.includes('mouse') ? 'mousemove' : 'touchmove'
            const endEvent = event.type.includes('mouse') ? 'mouseup' : 'touchend'

            document.addEventListener(moveEvent, moveHandler)
            document.addEventListener(endEvent, endHandler)

            // Store handlers for cleanup
            this._moveHandler = moveHandler
            this._endHandler = endHandler
            this._moveEvent = moveEvent
            this._endEvent = endEvent
        },

        handleMouseMove(event) {
            if (!this.isDragging) return
            this.dragX = event.clientX - this.startX
            this.dragY = event.clientY - this.startY
        },

        handleTouchMove(event) {
            if (!this.isDragging) return
            this.dragX = event.touches[0].clientX - this.startX
            this.dragY = event.touches[0].clientY - this.startY
        },

        handleMouseUp() {
            this.finalizeDrag()
            document.removeEventListener(this._moveEvent, this._moveHandler)
            document.removeEventListener(this._endEvent, this._endHandler)
        },

        handleTouchEnd() {
            this.finalizeDrag()
            document.removeEventListener(this._moveEvent, this._moveHandler)
            document.removeEventListener(this._endEvent, this._endHandler)
        },

        finalizeDrag() {
            this.isDragging = false
            const threshold = 80

            if (Math.abs(this.dragX) > threshold) {
                if (this.dragX > 0) {
                    this.handleSwipe('like')
                } else {
                    this.handleSwipe('dislike')
                }
            } else if (Math.abs(this.dragX) < 10 && Math.abs(this.dragY) < 10) {
                // No significant movement - treat as click
                this.showDetails()
                this.dragX = 0
                this.dragY = 0
            } else {
                // Reset position if swipe didn't meet threshold
                this.dragX = 0
                this.dragY = 0
            }
        }
    },

    mounted() {
        this.loadRecommendations()
    }
}
</script>

<style scoped>
.recommendations-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 30px;
    padding: 40px 20px;
}

/* ===== EMPTY STATE ===== */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 24px;
    padding: 60px 20px;
}

.spinner {
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

.empty-icon {
    font-size: 60px;
}

.empty-state p {
    font-size: 18px;
    color: var(--text-secondary);
    margin: 0;
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

/* ===== CARD WRAPPER ===== */
.card-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    width: 100%;
    max-width: 900px;
    height: 500px;
}

/* ===== ARROW BUTTONS ===== */
.arrow-btn {
    width: 60px;
    height: 100%;
    background-color: var(--primary-purple-light);;
    border: none;
    border-radius: 12px;
    font-size: 32px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.arrow-btn:hover:not(:disabled) {
    background-color: rgba(216, 191, 216, 0.5);
    transform: scale(1.05);
}

.arrow-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.arrow-btn:hover:not(:disabled) {
    background-color: rgba(216, 191, 216, 0.4);
}

/* ===== BOOK CARD ===== */
.book-card {
    flex: 1;
    height: 100%;
    background: var(--primary-purple-dark);
    border-radius: 12px;
    padding: 30px;
    display: flex;
    gap: 30px;
    cursor: pointer;
    transition: transform 0.1s ease, opacity 0.1s ease;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    user-select: none;
}

.card-image {
    flex-shrink: 0;
}

.image-placeholder {
    width: 140px;
    height: 200px;
    background-color: rgba(255, 255, 255, 0.3);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 64px;
    color: white;
}

.card-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.card-title {
    font-size: 28px;
    font-weight: 700;
    color: white;
    margin: 0 0 12px 0;
    line-height: 1.3;
}

.card-authors {
    font-size: 16px;
    color: rgba(255, 255, 255, 0.9);
    margin: 0 0 24px 0;
}

.card-score {
    font-size: 18px;
    font-weight: 600;
    color: white;
}

/* ===== DRAG OVERLAY ===== */
.drag-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 12px;
}

.swipe-indicator {
    font-size: 36px;
    font-weight: 700;
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.swipe-indicator.like {
    color: #8FD3A8;
}

.swipe-indicator.dislike {
    color: #F8A8A8;
}

/* ===== PROGRESS ===== */
.progress-section {
    width: 100%;
    max-width: 500px;
}

.progress-text {
    text-align: center;
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.progress-bar {
    width: 100%;
    height: 6px;
    background-color: var(--border-color);
    border-radius: 3px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--primary-purple);
    transition: width 0.3s ease;
}

/* ===== INSTRUCTIONS ===== */
.instructions {
    display: flex;
    justify-content: space-around;
    width: 100%;
    max-width: 500px;
    font-size: 14px;
    color: var(--text-secondary);
    margin-top: 20px;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .card-wrapper {
        height: 400px;
        gap: 12px;
    }

    .arrow-btn {
        width: 50px;
        font-size: 24px;
    }

    .book-card {
        flex-direction: column;
        padding: 20px;
    }

    .card-image {
        width: 100%;
        display: flex;
        justify-content: center;
    }

    .image-placeholder {
        width: 100px;
        height: 150px;
        font-size: 48px;
    }

    .card-title {
        font-size: 22px;
    }

    .card-authors {
        font-size: 14px;
    }
}

@media (max-width: 480px) {
    .recommendations-container {
        padding: 20px;
        gap: 20px;
    }

    .card-wrapper {
        flex-direction: column;
        height: auto;
        gap: 12px;
    }

    .arrow-btn {
        width: 100%;
        height: 50px;
        font-size: 24px;
    }

    .arrow-left,
    .arrow-right {
        flex-direction: row;
    }

    .book-card {
        gap: 20px;
    }
}
</style>
