# Email Subscription Modal Redesign - Newneek-inspired UX

## Summary

Replaced the 3-step email subscription modal with a modern, Newneek-inspired single-screen design that improves user experience and reduces friction.

## Changes Made

### 1. JavaScript Functions (Lines 12788-13162)

**Removed Functions:**
- `resetAlertModal()`, `loadAlertSettings()`, `toggleAlertOptions()`
- `goToAlertStep()`, `onEmailInputChange()`, `sendVerificationEmail()`
- `saveAlertSettings()`, `revokeConsent()`

**New Functions:**
- `showSubscribeScreen(screenName)` - Screen management
- `toggleAlertCard(type)` - Card-checkbox bidirectional binding
- `syncCardFromCheckbox(type)` - Sync card from checkbox change
- `startSubscription()` - Main subscription flow
- `showAlreadyRegistered(settings)` - Show already registered popup
- `goToEditSettings()` - Navigate to edit screen
- `loadEditSettings(email, alertTypes)` - Load edit screen
- `saveEditedSettings()` - Save edited settings
- `unsubscribe()` - Unsubscribe with confirmation
- `goBackToMain()` - Navigate back to main screen
- `resendVerification()` - Resend verification email

**Retained Functions:**
- `openAlertSettings()`, `closeAlertSettings()` (refactored)
- `startVerificationPolling()`, `stopVerificationPolling()`, `onVerificationComplete()`
- `checkEmailVerification()`, `verifyEmailToken()` (legacy compatibility)

### 2. Modal HTML (Lines 13166-13349 → 13166-13511)

**New Screens:**

1. **subscribe-main** (Default)
   - Header with title and subtitle
   - 3 interactive alert type cards (rank_change, important_insight, daily_summary)
   - Cards are clickable and toggle selection
   - Bidirectional sync with checkbox list below
   - Email input field
   - Big CTA button: "🔔 무료 구독 시작하기"
   - Footer notes about consent

2. **subscribe-already-registered** (Popup overlay)
   - Shown when email is already registered
   - Offers to edit settings or close
   - Overlay with backdrop blur

3. **subscribe-edit** (Settings management)
   - Back button to return to main
   - Shows current email
   - Toggle switches for each alert type
   - Save and Unsubscribe buttons

4. **subscribe-verification** (Email verification)
   - Envelope icon
   - Shows email address
   - 30-minute timeout notice
   - Resend button
   - Loading spinner with "인증 대기중..."

5. **subscribe-complete** (Success)
   - Party popper icon
   - Summary of email and selected alerts
   - Confirmation message
   - Close button

### 3. CSS Styles (Lines 13351-13424 → 13425-13511)

**New Styles:**
- `.alert-card` - Card styling with hover and selected states
- `.alert-card.selected` - Blue border and background when selected
- `.alert-card:hover` - Hover effect with shadow
- `.toggle-switch` - Custom toggle switch component
- `.toggle-slider` - Toggle slider with smooth transitions
- `@keyframes spin` - Spinner animation for verification screen
- Responsive breakpoints for mobile (< 640px)

### 4. Design System Compliance

All styles use AMOREPACIFIC color palette:
- `--pacific-blue: #001C58` (headers, primary)
- `--amore-blue: #1F5795` (accents, links, borders)
- Gray: `#64748b`, `#94a3b8`, `#cbd5e1`, `#e2e8f0`, `#f1f5f9`, `#f8fafc`
- Success: `#10b981`
- Warning: `#f59e0b`
- Error: `#dc2626`, `#fee2e2`

### 5. API Integration

**New Endpoints Used:**
- `POST /api/v4/subscribe` - Initial subscription with email + alert_types
- `GET /api/v4/alert-settings?email=...` - Get current settings
- `PUT /api/v4/alert-settings` - Update settings
- `DELETE /api/v4/alert-settings?email=...` - Unsubscribe

**Existing Endpoints:**
- `GET /api/alerts/verification-status?email=...` - Polling (unchanged)
- `POST /api/alerts/send-verification` - Resend email (unchanged)
- `POST /api/alerts/verify-email` - Legacy token verification (unchanged)

## Key UX Improvements

1. **Single-screen approach** - All options visible at once (inspired by Newneek)
2. **Visual card selection** - More engaging than plain checkboxes
3. **Bidirectional sync** - Cards and checkboxes stay in sync
4. **Clear hierarchy** - Main → Already Registered → Edit OR Verification → Complete
5. **Progressive disclosure** - Only show verification/edit when needed
6. **Mobile-responsive** - Cards stack vertically on small screens

## Testing Checklist

- [ ] Open modal - should show main screen with 3 cards
- [ ] Click card - should toggle selection and sync checkbox
- [ ] Click checkbox - should toggle card selection
- [ ] Subscribe with new email - should show verification screen
- [ ] Subscribe with existing email - should show already registered popup
- [ ] Edit settings - should load current settings with toggles
- [ ] Save settings - should update and close
- [ ] Unsubscribe - should show confirmation and delete
- [ ] Verification polling - should auto-advance to complete screen
- [ ] Resend verification - should work without error
- [ ] Mobile view - cards should stack vertically

## File Changes

- `dashboard/amore_unified_dashboard_v4.html`
  - JavaScript: ~470 lines replaced
  - HTML: ~180 lines replaced
  - CSS: ~90 lines replaced
  - Total: ~740 lines modified

## Line Count Change

- Before: 13,426 lines
- After: 13,513 lines
- Net change: +87 lines
