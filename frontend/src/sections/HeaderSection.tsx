import { useTranslation } from "react-i18next";

export default function HeaderSection() {
  const { t } = useTranslation();

  return (
    <header className="relative flex flex-col items-center gap-4 px-4 pb-6 pt-12 text-center">
      <p className="text-sm tracking-[0.3em] text-ob-muted uppercase">
        {t("header.tagline")}
      </p>
      <h1 className="gradient-title text-5xl font-bold leading-tight md:text-6xl">
        {t("header.title")}
      </h1>
      <p className="max-w-2xl text-base text-ob-muted md:text-lg">
        {t("header.subtitle")}
      </p>
    </header>
  );
}
