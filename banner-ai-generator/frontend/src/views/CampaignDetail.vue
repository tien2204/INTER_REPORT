<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <button
          @click="$router.go(-1)"
          class="text-gray-400 hover:text-gray-500"
        >
          <ArrowLeftIcon class="h-6 w-6" />
        </button>
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Campaign Details</h1>
          <p class="mt-1 text-sm text-gray-500">
            View campaign information and related banners
          </p>
        </div>
      </div>
    </div>

    <!-- Placeholder content -->
    <div class="card text-center py-12">
      <h3 class="text-lg font-medium text-gray-900 mb-2">Campaign Detail View</h3>
      <p class="text-gray-500">
        This view will show detailed campaign information and associated banners.
      </p>
      <p class="text-sm text-gray-400 mt-2">
        Campaign ID: {{ $route.params.id }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ArrowLeftIcon } from '@heroicons/vue/24/outline'
</script>
