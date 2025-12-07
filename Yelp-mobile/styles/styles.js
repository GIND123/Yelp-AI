// styles.js
import { StyleSheet, Platform } from 'react-native';

export const COLORS = {
  // === Base Palette (Shared) ===
  primary: '#D1252A', // Main Green
  backgroundLight: '#F9F9F9',
  textLight: '#1F2937', // Dark Gray/Black for headings
  textSecondary: '#6B7280', // Gray for subtitles
  surfaceLight: '#FFFFFF',
  shadow: '#E5E7EB', // gray-200
  borderColor: '#CBD5E1', // slate-300
  gold: '#F59E0B', // For stars

  // === Specific Accents (Added for Summary Screen) ===
  backgroundDark: '#102213',
  white: '#FFFFFF',

  // Summary Block Colors
  accentGreen: '#4CAF50',
  accentRedBg: '#FFEBEE',
  accentRedText: '#D32F2F',
  accentBlue: '#2196F3',
  accentBlueBg: '#E3F2FD',
};

export const styles = StyleSheet.create({
  // =========================
  // 1. GLOBAL LAYOUT
  // =========================
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.backgroundLight,
  },
  container: {
    flex: 1,
    width: '95%',
    alignSelf: 'center',
    paddingTop: 15,
    backgroundColor: COLORS.backgroundLight,
  },

  // =========================
  // 2. HEADERS (SHARED)
  // =========================
  header: {
    height: 80,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    backgroundColor: 'rgba(249, 249, 249, 0.9)',
    borderBottomWidth: 1,
    borderBottomColor: 'transparent',
    zIndex: 10,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: COLORS.textLight,
  },
  navTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.textLight,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerSpacer: {
    width: 40,
  },
  profileButton: {
    height: 48,
    width: 48,
    borderRadius: 24,
    overflow: 'hidden',
  },
  profileImage: {
    height: '100%',
    width: '100%',
    resizeMode: 'cover',
  },

  // =========================
  // 3. HOME FEED CARDS
  // =========================
  scrollContent: {
    paddingHorizontal: 24,
    paddingTop: 16,
    paddingBottom: 120,
  },
  cardContainer: {
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 24,
  },
  cardImageContainer: {
    height: 200,
    width: '100%',
  },
  cardImage: {
    height: '100%',
    width: '100%',
    resizeMode: 'cover',
  },
  cardContent: {
    padding: 20,
    gap: 8,
  },

  // =========================
  // 4. SHARED TYPOGRAPHY (Lists/Feed)
  // =========================
  restaurantName: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.textLight,
    marginBottom: 4,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 6,
  },
  metaText: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginLeft: 4,
  },
  dotSeparator: {
    marginHorizontal: 6,
    color: '#D1D5DB',
  },
  cuisineText: {
    fontSize: 14,
    color: COLORS.textSecondary,
  },
  ratingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 8,
  },
  ratingText: {
    marginLeft: 2,
    fontSize: 14,
    fontWeight: '500',
    color: COLORS.gold,
  },

  // =========================
  // 5. SCREEN: RESTAURANT SELECTION LIST
  // =========================
  restaurantListContent: {
    padding: 24,
    paddingBottom: 120,
  },
  restaurantCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    gap: 16,
  },
  restaurantCardSelected: {
    backgroundColor: 'rgba(76, 175, 80, 0.08)', // Primary 8% opacity
    borderWidth: 2,
    borderColor: COLORS.primary,
    // iOS shadow
    ...Platform.select({
      ios: {
        shadowColor: COLORS.primary,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.15,
        shadowRadius: 4,
      },
      android: {
        elevation: 0, // Remove elevation on Android to avoid dark shadow
      },
    }),
  },
  restaurantCardUnselected: {
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    // iOS shadow
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
      },
      android: {
        elevation: 0, // Remove elevation on Android
      },
    }),
  },
  restaurantThumbnail: {
    width: 64,
    height: 64,
    borderRadius: 8,
    backgroundColor: COLORS.shadow,
  },
  selectionRadio: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  selectionRadioSelected: {
    backgroundColor: COLORS.primary,
    borderWidth: 2,
    borderColor: COLORS.primary,
  },
  selectionRadioUnselected: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: COLORS.shadow,
  },

  // =========================
  // 6. SCREEN: RESTAURANT SUMMARY DETAIL
  // =========================

  // Hero Image
  heroImageContainer: {
    position: 'relative',
    width: '100%',
    height: 320,
  },
  heroImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },

  // The Floating Phone Button
  floatingCallBtn: {
    position: 'absolute',
    bottom: -28,
    right: 32,
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: COLORS.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },

  // Summary Content Layout
  summaryContentPadding: {
    padding: 16,
    paddingTop: 20,
  },
  restaurantTitleLarge: {
    fontSize: 32,
    fontWeight: '700',
    color: COLORS.textLight,
    marginBottom: 4,
    letterSpacing: -0.5,
  },
  cuisineTextLarge: {
    fontSize: 16,
    color: COLORS.textSecondary,
    marginBottom: 12,
  },

  // Detailed Rating Row
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  starsContainer: {
    flexDirection: 'row',
    marginRight: 8,
  },
  ratingNumberLarge: {
    fontSize: 16,
    fontWeight: '700',
    color: COLORS.textLight,
    marginRight: 4,
  },
  reviewCount: {
    fontSize: 14,
    color: COLORS.textSecondary,
  },

  // Good/Bad/Info Blocks
  summaryCardBlock: {
    borderRadius: 12,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  summaryBlockTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 16,
  },
  summaryListContainer: {
    gap: 12,
  },
  summaryListItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  summaryListText: {
    flex: 1,
    fontSize: 16,
    lineHeight: 24,
    color: COLORS.textLight,
  },
  summaryDisclaimer: {
    textAlign: 'center',
    fontSize: 12,
    color: COLORS.textSecondary,
    marginTop: 8,
    marginBottom: 16,
  },

  // =========================
  // 7. SCREEN: CAMERA / UPLOAD
  // =========================
  camera: {
    flex: 1,
  },
  uploadPlaceholder: {
    width: '100%',
    aspectRatio: 1,
    borderWidth: 2,
    borderColor: COLORS.borderColor,
    borderStyle: 'dashed',
    borderRadius: 12,
    backgroundColor: 'rgba(241, 245, 249, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
    gap: 16,
    overflow: 'hidden',
  },
  uploadedImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  uploadPlaceholderTextTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.textLight,
  },
  uploadPlaceholderTextSub: {
    fontSize: 14,
    color: COLORS.textSecondary,
  },
  buttonGrid: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 24,
  },
  inputContainer: {
    marginBottom: 24,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#475569',
    marginBottom: 8,
  },
  textArea: {
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: COLORS.borderColor,
    borderRadius: 12,
    padding: 12,
    fontSize: 16,
    color: COLORS.textLight,
    height: 120,
    textAlignVertical: 'top',
  },

  // =========================
  // 8. SHARED FOOTER & BUTTONS
  // =========================
  bottomFixedFooter: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 24,
    paddingBottom: Platform.OS === 'ios' ? 40 : 24,
    backgroundColor: COLORS.backgroundLight,
    borderTopWidth: 1,
    borderTopColor: 'rgba(229, 231, 235, 0.5)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 8,
  },
  btnBase: {
    height: 56,
    borderRadius: 28,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 30,
  },
  btnOutline: {
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: COLORS.borderColor,
    flex: 1,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    height: 56,
  },
  btnPrimary: {
    backgroundColor: COLORS.primary,
    width: '100%',
    borderRadius: 28,
    height: 56,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  btnTextOutline: {
    fontSize: 16,
    fontWeight: '600',
    color: '#334155',
  },
  btnTextPrimary: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  fabContainer: {
    position: 'absolute',
    bottom: 24,
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  fabButton: {
    backgroundColor: COLORS.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 64,
    paddingHorizontal: 32,
    borderRadius: 32,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  fabText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },

  // =========================
  // 9. SCREEN: SCHEDULE ORDER (NEW)
  // =========================
  sectionContainer: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: COLORS.textLight,
    marginBottom: 12,
    marginLeft: 4,
  },

  // Hero Card / Image Display
  itemCard: {
    width: '100%',
    height: 220,
    backgroundColor: COLORS.surfaceLight,
    borderRadius: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
    borderWidth: 1,
    borderColor: COLORS.borderColor,
  },
  itemImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  itemTextContainer: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: COLORS.backgroundLight,
  },
  itemDescriptionText: {
    fontSize: 20,
    fontWeight: '600',
    color: COLORS.primary,
    textAlign: 'center',
    lineHeight: 28,
  },

  // Form Inputs
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1,
    borderColor: COLORS.borderColor,
    borderRadius: 12,
    paddingHorizontal: 16,
    height: 56,
  },
  textInput: {
    flex: 1,
    fontSize: 16,
    color: COLORS.textLight,
    height: '100%',
  },
  locationButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    paddingVertical: 8,
    gap: 8,
  },

  // Date/Time Layouts
  iosPickerContainer: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
  },
  androidRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  dateText: {
    fontSize: 15,
    color: COLORS.textLight,
  },
});
