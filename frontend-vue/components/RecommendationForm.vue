<template>
    <div class="search-panel-content">
        <h2 class="panel-title">Find Your Next Read</h2>

        <div class="form-group">
            <label for="userId" class="form-label">
                <span class="label-text">User Profile</span>
                <span class="label-hint">(optional)</span>
            </label>
            <input id="userId" :value="userId" type="text" class="form-input"
                placeholder="Enter your user ID or leave empty for discovery mode"
                @input="$emit('update:userId', $event.target.value)" :disabled="loading" />
        </div>

        <div class="form-grid">
            <div class="form-group">
                <label for="seedBookIds" class="form-label">
                    <span class="label-text">Seed Books</span>
                    <span class="label-hint">(optional)</span>
                </label>
                <input id="seedBookIds" :value="seedBookIds" type="text" class="form-input"
                    placeholder="e.g., 101, 202, 303" @input="$emit('update:seedBookIds', $event.target.value)"
                    :disabled="loading" />
                <p class="form-help">Enter comma-separated book IDs to use as inspiration</p>
            </div>

            <div class="form-group">
                <label for="k" class="form-label">
                    <span class="label-text">Number of Recommendations</span>
                </label>
                <div class="input-with-slider">
                    <input id="k" :value="k" type="range" min="1" max="50" class="form-slider"
                        @input="$emit('update:k', parseInt($event.target.value))" :disabled="loading" />
                    <span class="slider-value">{{ k }}</span>
                </div>
            </div>
        </div>

        <button @click="$emit('submit')" :disabled="loading" class="btn btn-primary">
            <span v-if="!loading" class="btn-icon">🔍</span>
            <span v-else class="spinner"></span>
            {{ loading ? 'Searching...' : 'Get Recommendations' }}
        </button>
    </div>
</template>

<script>
export default {
    props: {
        userId: { type: String, default: '' },
        seedBookIds: { type: String, default: '' },
        k: { type: Number, default: 3 },
        loading: { type: Boolean, default: false }
    },
    emits: ['submit', 'update:userId', 'update:seedBookIds', 'update:k']
}
</script>
