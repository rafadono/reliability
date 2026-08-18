// Sidebar duplicates the top tab bar's navigation today (per-card anchors that
// mostly overlap with just switching tabs). Kept behind a flag instead of
// deleted: those anchors are the natural place to expand into once a tab grows
// enough cards to need its own table of contents again. Set VITE_SHOW_SIDEBAR=true
// (see docker-compose.yml) to bring it back without reverting any code.
export const sidebarEnabled = import.meta.env.VITE_SHOW_SIDEBAR === 'true'
