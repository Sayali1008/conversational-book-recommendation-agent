<template>
    <div class="search-panel-content">
        <h2 class="panel-title">Find Your Next Read</h2>

        <!-- <div class="form-group">
            <label for="seedBookIds" class="form-label">
                <span class="label-text">Seed Books</span>
                <span class="label-hint">(optional)</span>
            </label>
            <input id="seedBookIds" :value="seedBookIds" type="text" class="form-input"
                placeholder="e.g., 101, 202, 303" @input="$emit('update:seedBookIds', $event.target.value)"
                :disabled="loading" />
            <p class="form-help">Enter comma-separated book IDs to use as inspiration</p>
        </div> -->

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
        seedBookIds: { type: String, default: '' },
        loading: { type: Boolean, default: false }
    },
    emits: ['submit', 'update:seedBookIds']
}
</script>
