import { Route, Routes } from "react-router";
import StarfieldBackdrop from "@/components/StarfieldBackdrop";
import Home from "@/pages/Home";

export default function App() {
  return (
    <>
      <StarfieldBackdrop />
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </>
  );
}
