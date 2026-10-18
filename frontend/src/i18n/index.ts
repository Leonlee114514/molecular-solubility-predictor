import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import zh from "./zh.json";
import en from "./en.json";

i18n.use(initReactI18next).init({
  resources: {
    zh: { translation: zh },
    en: { translation: en },
  },
  lng: "zh",
  fallbackLng: "zh",
  // Keys are flat dot-keys ("app.title" as a literal key), not nested objects.
  keySeparator: false,
  nsSeparator: false,
  interpolation: {
    escapeValue: false,
    // Backend format placeholders use Python style: {name}, {best}
    prefix: "{",
    suffix: "}",
  },
});

/* Keep <html lang> in sync with the active UI language (WCAG 3.1.1: the page
   language must be programmatically determinable). Without this the document
   keeps whatever index.html declared even after switching to the other
   language, so screen readers announce Chinese text with an English voice. */
const applyHtmlLang = (lng: string) => {
  document.documentElement.lang = lng.startsWith("zh") ? "zh-CN" : "en";
};

applyHtmlLang(i18n.language);
i18n.on("languageChanged", applyHtmlLang);

export default i18n;
