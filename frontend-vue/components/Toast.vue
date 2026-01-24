<template>
  <div v-if="show" class="toast" :class="type">
    <span class="toast-icon">{{ icon }}</span>
    <span>{{ message }}</span>
  </div>
</template>

<script>
export default {
  props: {
    message: { type: String, default: '' },
    type: { type: String, default: 'success' },
    icon: { type: String, default: '✓' },
    show: { type: Boolean, default: false }
  }
}
</script>
