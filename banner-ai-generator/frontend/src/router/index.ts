import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

// Import views
import Dashboard from '@/views/Dashboard.vue'
import CreateBanner from '@/views/CreateBanner.vue'
import Campaigns from '@/views/Campaigns.vue'
import SystemMonitor from '@/views/SystemMonitor.vue'
import BannerDetail from '@/views/BannerDetail.vue'
import CampaignDetail from '@/views/CampaignDetail.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Dashboard',
    component: Dashboard,
    meta: {
      title: 'Dashboard'
    }
  },
  {
    path: '/create',
    name: 'CreateBanner',
    component: CreateBanner,
    meta: {
      title: 'Create Banner'
    }
  },
  {
    path: '/campaigns',
    name: 'Campaigns',
    component: Campaigns,
    meta: {
      title: 'Campaigns'
    }
  },
  {
    path: '/campaigns/:id',
    name: 'CampaignDetail',
    component: CampaignDetail,
    meta: {
      title: 'Campaign Detail'
    }
  },
  {
    path: '/banners/:id',
    name: 'BannerDetail', 
    component: BannerDetail,
    meta: {
      title: 'Banner Detail'
    }
  },
  {
    path: '/monitor',
    name: 'SystemMonitor',
    component: SystemMonitor,
    meta: {
      title: 'System Monitor'
    }
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFound.vue'),
    meta: {
      title: 'Page Not Found'
    }
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    } else {
      return { top: 0 }
    }
  }
})

// Global navigation guard for title updates
router.beforeEach((to, from, next) => {
  // Update document title
  const title = to.meta?.title as string
  if (title) {
    document.title = `${title} | AI Banner Generator`
  }
  
  next()
})

export default router
